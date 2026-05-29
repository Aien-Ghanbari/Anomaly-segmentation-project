# Copyright (c) OpenMMLab. All rights reserved.
import csv
import glob
import importlib
import os
import os.path as osp
import random
import re
import sys
import warnings

import numpy as np
import torch
import yaml
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ood_metrics import fpr_at_95_tpr
from PIL import Image
from sklearn.metrics import average_precision_score
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, ToTensor

seed = 42

# General reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19

# GPU related behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


def sanitize_input_pattern(pattern):
    """Fix wildcard typo like `images\\.*png` -> `images\\*.png`."""
    fixed = re.sub(r"([/\\])\.\*", r"\1*", str(pattern))
    return os.path.expanduser(fixed)


def infer_dataset_name(path):
    """Infer dataset name from path for grouped reporting."""
    known_datasets = (
        "RoadAnomaly21",
        "RoadAnomaly",
        "RoadObsticle21",
        "fs_static",
        "FS_LostFound_full",
        "Streethazard",
    )
    lower_path = path.lower()
    for dataset_name in known_datasets:
        if dataset_name.lower() in lower_path:
            return dataset_name
    return "Unknown"


def configure_import_paths():
    """Ensure dynamic imports from EoMT project work from any CWD."""
    script_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.abspath(osp.join(script_dir, ".."))
    eomt_root = osp.join(project_root, "eomt")

    for path in (project_root, eomt_root):
        if osp.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def load_eomt_model(config_path, device, img_size=(512, 1024)):
    """Load EoMT model from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=NUM_CLASSES,
        encoder=encoder,
        **network_kwargs,
    )

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

    if name is None:
        warnings.warn("No logger name found in config. Initializing randomly.")
        model = lit_cls(
            img_size=img_size,
            num_classes=NUM_CLASSES,
            network=network,
            **model_kwargs,
        ).eval().to(device)
    else:
        try:
            state_dict_path = hf_hub_download(
                repo_id=f"tue-mps/{name}",
                filename="pytorch_model.bin",
            )
            is_dinov3 = "dinov3" in name
            if is_dinov3:
                model_kwargs["ckpt_path"] = state_dict_path
                model_kwargs["delta_weights"] = True

            model = lit_cls(
                img_size=img_size,
                num_classes=NUM_CLASSES,
                network=network,
                **model_kwargs,
            ).eval().to(device)

            if not is_dinov3:
                state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)

            print(f"EoMT model ({name}) loaded successfully from HuggingFace.")
        except RepositoryNotFoundError:
            warnings.warn(f"Pre-trained model not found for `{name}`.")
            model = lit_cls(
                img_size=img_size,
                num_classes=NUM_CLASSES,
                network=network,
                **model_kwargs,
            ).eval().to(device)

    return model


def normalize_map(score_map):
    """Min-max normalize to [0, 1] with epsilon-safe denominator."""
    min_v = np.min(score_map)
    max_v = np.max(score_map)
    denom = max(max_v - min_v, 1e-12)
    return (score_map - min_v) / denom


def compute_rba_map(result):
    """
    Compute RbA (Residual-based Anomaly) from per-pixel logits [C, H, W].

    RbA here combines:
    - confidence residual: 1 - max softmax probability,
    - entropy residual: normalized per-pixel entropy,
    - neighborhood residual: local variance of confidence (captures inconsistent regions).
    """
    probs = torch.softmax(result, dim=0)

    max_prob = torch.max(probs, dim=0).values
    conf_residual = 1.0 - max_prob

    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=0)
    entropy_norm = entropy / np.log(result.shape[0])

    conf_map = max_prob.unsqueeze(0).unsqueeze(0)
    local_mean = F.avg_pool2d(conf_map, kernel_size=7, stride=1, padding=3)
    local_mean_sq = F.avg_pool2d(conf_map * conf_map, kernel_size=7, stride=1, padding=3)
    local_var = (local_mean_sq - (local_mean * local_mean)).squeeze(0).squeeze(0)

    conf_residual_np = conf_residual.detach().cpu().numpy()
    entropy_np = entropy_norm.detach().cpu().numpy()
    local_var_np = local_var.detach().cpu().numpy()

    conf_residual_np = normalize_map(conf_residual_np)
    entropy_np = normalize_map(entropy_np)
    local_var_np = normalize_map(local_var_np)

    # Weighted residual fusion.
    rba_map = 0.5 * conf_residual_np + 0.3 * entropy_np + 0.2 * local_var_np
    return normalize_map(rba_map)


def compute_metrics(ood_gts, anomaly_scores):
    """Compute AUPRC and FPR@95 from flattened score maps."""
    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    if len(ood_out) == 0 or len(ind_out) == 0:
        return np.nan, np.nan

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    return prc_auc, fpr


def format_results_table(rows):
    headers = [
        "Dataset",
        "Images",
        "RbA AUPRC",
        "RbA FPR95",
    ]

    str_rows = []
    for row in rows:
        str_rows.append([
            row["dataset"],
            str(row["images"]),
            f"{row['RbA']['auprc'] * 100.0:.4f}" if not np.isnan(row["RbA"]["auprc"]) else "nan",
            f"{row['RbA']['fpr95'] * 100.0:.4f}" if not np.isnan(row["RbA"]["fpr95"]) else "nan",
        ])

    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row_cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row_cells))

    separator = "-+-".join("-" * w for w in widths)
    lines = [fmt(headers), separator]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def write_results_csv(rows, config_path, csv_path):
    fieldnames = [
        "Config",
        "Dataset",
        "Images",
        "RbA_AUPRC_percent",
        "RbA_FPR95_percent",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            auprc = row["RbA"]["auprc"]
            fpr95 = row["RbA"]["fpr95"]
            writer.writerow(
                {
                    "Config": config_path,
                    "Dataset": row["dataset"],
                    "Images": row["images"],
                    "RbA_AUPRC_percent": f"{auprc * 100.0:.4f}" if not np.isnan(auprc) else "nan",
                    "RbA_FPR95_percent": f"{fpr95 * 100.0:.4f}" if not np.isnan(fpr95) else "nan",
                }
            )


def main():
    configure_import_paths()

    parser = ArgumentParser()
    default_config = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "..",
            "eomt",
            "configs",
            "dinov2",
            "cityscapes",
            "semantic",
            "eomt_base_640.yaml",
        )
    )

    parser.add_argument(
        "--input",
        default=[
            r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\Validation_Dataset\RoadAnomaly\images\*.jpg",
            r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\Validation_Dataset\RoadAnomaly21\images\*.png",
            r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\Validation_Dataset\RoadAnomaly\images\*.jpg",
            r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\Validation_Dataset\fs_static\images\*.jpg",
            r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\Validation_Dataset\FS_LostFound_full\images\.*png",
        ],
        nargs="+",
        help="A list of space separated input images",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the EoMT YAML config file",
    )
    parser.add_argument(
        "--output_csv",
        default=osp.abspath(osp.join(osp.dirname(__file__), "results_eomt_rba.csv")),
        help="Path to output CSV file (default: eval/results_eomt_rba.csv)",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not osp.isfile(args.config):
        parser.error(f"Config file not found: {args.config}. Pass a valid path with --config.")

    txt_output_path = osp.abspath(osp.join(osp.dirname(__file__), "results_eomt_rba.txt"))
    if not osp.exists(txt_output_path):
        open(txt_output_path, "w").close()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    print("Loading EoMT model from config: " + args.config)
    model = load_eomt_model(args.config, device, img_size=(512, 1024))

    supported_exts = [".jpg", ".jpeg", ".png", ".webp"]
    matched_paths = []
    for input_pattern in args.input:
        expanded_pattern = sanitize_input_pattern(input_pattern)
        direct_matches = glob.glob(expanded_pattern)

        if direct_matches:
            matched_paths.extend(direct_matches)
            continue

        base_pattern, ext = osp.splitext(expanded_pattern)
        if ext.lower() in supported_exts:
            for candidate_ext in supported_exts:
                candidate_pattern = base_pattern + candidate_ext
                matched_paths.extend(glob.glob(candidate_pattern))

    matched_paths = sorted(list(set(matched_paths)))
    if len(matched_paths) == 0:
        print("No input images matched the provided --input pattern(s).")
        write_results_csv([], args.config, args.output_csv)
        print(f"CSV saved to {args.output_csv}")
        return

    print(f"Matched {len(matched_paths)} input image(s).")

    per_dataset_data = {}

    for idx, path in enumerate(matched_paths, start=1):
        if idx % 25 == 0 or idx == 1 or idx == len(matched_paths):
            print(f"Processing image {idx}/{len(matched_paths)}")

        dataset_name = infer_dataset_name(path)
        if dataset_name not in per_dataset_data:
            per_dataset_data[dataset_name] = {
                "image_count": 0,
                "ood_gts": [],
                "rba_scores": [],
            }

        image_tensor = input_transform(Image.open(path).convert("RGB"))
        image_tensor = image_tensor.mul(255.0).clamp(0, 255).to(torch.uint8).to(device)
        image_batch = image_tensor.unsqueeze(0)

        with torch.no_grad(), autocast(
            dtype=torch.float16,
            device_type="cuda",
            enabled=(device.type == "cuda"),
        ):
            mask_logits_per_layer, class_logits_per_layer = model(image_batch)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1],
                image_tensor.shape[-2:],
                mode="bilinear",
            )
            per_pixel_logits = model.to_per_pixel_logits_semantic(
                mask_logits,
                class_logits_per_layer[-1],
            )
            result = per_pixel_logits[0]

        rba_map = compute_rba_map(result)

        path_gt = path.replace("images", "labels_masks")
        if "RoadObsticle21" in path_gt:
            path_gt = path_gt.replace("webp", "png")
        if "fs_static" in path_gt:
            path_gt = path_gt.replace("jpg", "png")
        if "RoadAnomaly" in path_gt and "RoadAnomaly21" not in path_gt:
            path_gt = path_gt.replace("jpg", "png")

        if not osp.exists(path_gt):
            warnings.warn(f"GT mask not found, skipping: {path_gt}")
            continue

        mask = Image.open(path_gt)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in path_gt and "RoadAnomaly21" not in path_gt:
            ood_gts = np.where(ood_gts == 2, 1, ood_gts)

        if "Streethazard" in path_gt:
            ood_gts = np.where(ood_gts == 14, 255, ood_gts)
            ood_gts = np.where(ood_gts < 20, 0, ood_gts)
            ood_gts = np.where(ood_gts == 255, 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue

        per_dataset_data[dataset_name]["ood_gts"].append(ood_gts)
        per_dataset_data[dataset_name]["rba_scores"].append(rba_map)
        per_dataset_data[dataset_name]["image_count"] += 1

        del result, rba_map, ood_gts, mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    all_ood_gts = []
    all_rba_scores = []

    for dataset_name in sorted(per_dataset_data.keys()):
        entry = per_dataset_data[dataset_name]
        if not entry["ood_gts"]:
            continue

        ood_gts = np.array(entry["ood_gts"])
        rba_scores = np.array(entry["rba_scores"])
        prc_auc, fpr = compute_metrics(ood_gts, rba_scores)

        rows.append(
            {
                "dataset": dataset_name,
                "images": entry["image_count"],
                "RbA": {"auprc": prc_auc, "fpr95": fpr},
            }
        )

        all_ood_gts.append(ood_gts)
        all_rba_scores.append(rba_scores)

    if not rows:
        print("No valid OOD samples were found.")
        write_results_csv([], args.config, args.output_csv)
        print(f"CSV saved to {args.output_csv}")
        return

    concat_ood_gts = np.concatenate(all_ood_gts, axis=0)
    concat_rba_scores = np.concatenate(all_rba_scores, axis=0)
    all_prc_auc, all_fpr = compute_metrics(concat_ood_gts, concat_rba_scores)

    rows.append(
        {
            "dataset": "ALL",
            "images": int(sum(r["images"] for r in rows)),
            "RbA": {"auprc": all_prc_auc, "fpr95": all_fpr},
        }
    )

    table = format_results_table(rows)
    print("\nEvaluation Summary (RbA AUPRC/FPR@TPR95 in %)")
    print(table)

    write_results_csv(rows, args.config, args.output_csv)
    print(f"CSV saved to {args.output_csv}")

    with open(txt_output_path, "a") as file:
        file.write(f"Config: {args.config}\n")
        file.write("Evaluation Summary (RbA AUPRC/FPR@TPR95 in %)\n")
        file.write(table + "\n\n")


if __name__ == "__main__":
    main()
