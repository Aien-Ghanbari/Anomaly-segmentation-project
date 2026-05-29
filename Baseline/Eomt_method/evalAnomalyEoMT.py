# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import csv
import glob
import sys
import torch
import random
import re
import yaml
import importlib
import warnings
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

# --- NEW IMPORTS FOR EoMT ---
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
# Cityscapes usually has 19 known classes for evaluation
NUM_CLASSES = 19 

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


METHODS = ("MSP", "MaxLogit", "MaxEntropy")


def sanitize_input_pattern(pattern):
    """Fix common wildcard typo like `images\\.*png` -> `images\\*.png`."""
    fixed = re.sub(r"([/\\])\.\*", r"\1*", str(pattern))
    return os.path.expanduser(fixed)


def infer_dataset_name(path):
    """Infer dataset name from the image path for per-dataset reporting."""
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


def compute_anomaly_maps(result):
    """Compute all anomaly score maps from per-pixel logits [C, H, W]."""
    result_np = result.detach().cpu().numpy()
    msp_map = 1.0 - np.max(result_np, axis=0)
    maxlogit_map = -np.max(result_np, axis=0)

    probs = torch.nn.functional.softmax(result, dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=0)
    maxentropy_map = entropy.detach().cpu().numpy()

    return {
        "MSP": msp_map,
        "MaxLogit": maxlogit_map,
        "MaxEntropy": maxentropy_map,
    }


def compute_metrics(ood_gts, anomaly_scores):
    """Compute AUPRC and FPR@95 from flattened score maps."""
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

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
        "MSP AUPRC",
        "MSP FPR95",
        "MaxLogit AUPRC",
        "MaxLogit FPR95",
        "MaxEntropy AUPRC",
        "MaxEntropy FPR95",
    ]

    str_rows = []
    for row in rows:
        str_rows.append([
            row["dataset"],
            str(row["images"]),
            f"{row['MSP']['auprc'] * 100.0:.4f}",
            f"{row['MSP']['fpr95'] * 100.0:.4f}",
            f"{row['MaxLogit']['auprc'] * 100.0:.4f}",
            f"{row['MaxLogit']['fpr95'] * 100.0:.4f}",
            f"{row['MaxEntropy']['auprc'] * 100.0:.4f}",
            f"{row['MaxEntropy']['fpr95'] * 100.0:.4f}",
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
    """Write summary rows to CSV using the same metrics shown in the table."""
    fieldnames = [
        "Config",
        "Dataset",
        "Images",
        "MSP_AUPRC_percent",
        "MSP_FPR95_percent",
        "MaxLogit_AUPRC_percent",
        "MaxLogit_FPR95_percent",
        "MaxEntropy_AUPRC_percent",
        "MaxEntropy_FPR95_percent",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Config": config_path,
                    "Dataset": row["dataset"],
                    "Images": row["images"],
                    "MSP_AUPRC_percent": f"{row['MSP']['auprc'] * 100.0:.4f}",
                    "MSP_FPR95_percent": f"{row['MSP']['fpr95'] * 100.0:.4f}",
                    "MaxLogit_AUPRC_percent": f"{row['MaxLogit']['auprc'] * 100.0:.4f}",
                    "MaxLogit_FPR95_percent": f"{row['MaxLogit']['fpr95'] * 100.0:.4f}",
                    "MaxEntropy_AUPRC_percent": f"{row['MaxEntropy']['auprc'] * 100.0:.4f}",
                    "MaxEntropy_FPR95_percent": f"{row['MaxEntropy']['fpr95'] * 100.0:.4f}",
                }
            )


def configure_import_paths():
    """Ensure dynamic imports from the EoMT project work from any CWD."""
    script_dir = osp.dirname(osp.abspath(__file__))
    project_root = osp.abspath(osp.join(script_dir, ".."))
    eomt_root = osp.join(project_root, "eomt")

    for path in (project_root, eomt_root):
        if osp.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

def load_eomt_model(config_path, device, img_size=(512, 1024)):
    """Helper function to load the EoMT model from its YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # Load network
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

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    
    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

    if name is None:
        warnings.warn("No logger name found in the config. Initializing randomly.")
        model = lit_cls(img_size=img_size, num_classes=NUM_CLASSES, network=network, **model_kwargs).eval().to(device)
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

            model = lit_cls(img_size=img_size, num_classes=NUM_CLASSES, network=network, **model_kwargs).eval().to(device)

            if not is_dinov3:
                state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                
            print(f"EoMT Model ({name}) LOADED successfully from HuggingFace.")
        except RepositoryNotFoundError:
            warnings.warn(f"Pre-trained model not found for `{name}`.")
            model = lit_cls(img_size=img_size, num_classes=NUM_CLASSES, network=network, **model_kwargs).eval().to(device)
            
    return model

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
        '--config',
        default=default_config,
        help="Path to the EoMT YAML config file (default: eomt_base_640.yaml)",
    )
    parser.add_argument(
        "--output_csv",
        default=osp.abspath(osp.join(osp.dirname(__file__), "results_eomt.csv")),
        help="Path to output CSV file (default: eval/results_eomt.csv)",
    )
    parser.add_argument('--datadir', default=r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\Validation_Dataset")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if not osp.isfile(args.config):
        parser.error(
            f"Config file not found: {args.config}. Pass a valid path with --config."
        )
    
    per_dataset_data = {}

    txt_output_path = osp.abspath(osp.join(osp.dirname(__file__), "results_eomt.txt"))
    if not os.path.exists(txt_output_path):
        open(txt_output_path, 'w').close()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    print ("Loading EoMT model from config: " + args.config)
    model = load_eomt_model(args.config, device, img_size=(512, 1024))
    
    supported_exts = [".jpg", ".jpeg", ".png", ".webp"]
    matched_paths = []
    tried_patterns = []
    for input_pattern in args.input:
        expanded_pattern = sanitize_input_pattern(input_pattern)
        direct_matches = glob.glob(expanded_pattern)
        tried_patterns.append(expanded_pattern)

        if direct_matches:
            matched_paths.extend(direct_matches)
            continue

        base_pattern, ext = osp.splitext(expanded_pattern)
        if ext.lower() in supported_exts:
            for candidate_ext in supported_exts:
                candidate_pattern = base_pattern + candidate_ext
                tried_patterns.append(candidate_pattern)
                matched_paths.extend(glob.glob(candidate_pattern))

    matched_paths = sorted(list(set(matched_paths)))

    if len(matched_paths) == 0:
        print("No input images matched the provided --input pattern(s).")
        return

    print(f"Matched {len(matched_paths)} input image(s).")

    for idx, path in enumerate(matched_paths, start=1):
            if idx % 25 == 0 or idx == 1 or idx == len(matched_paths):
                print(f"Processing image {idx}/{len(matched_paths)}")
            dataset_name = infer_dataset_name(path)

            if dataset_name not in per_dataset_data:
                per_dataset_data[dataset_name] = {
                    "image_count": 0,
                    "ood_gts": [],
                    "scores": {method: [] for method in METHODS},
                }
            
            # 1. Prepare Image ([C,H,W] uint8, then batch to [B,C,H,W])
            image_tensor = input_transform(Image.open(path).convert('RGB'))
            image_tensor = (
                image_tensor.mul(255.0)
                .clamp(0, 255)
                .to(torch.uint8)
                .to(device)
            )
            image_batch = image_tensor.unsqueeze(0)

            # 2. EoMT Forward Pass (direct full-image inference)
            with torch.no_grad(), autocast(
                dtype=torch.float16,
                device_type="cuda",
                enabled=(device.type == "cuda"),
            ):
                # Get raw mask and class logits from the transformer
                mask_logits_per_layer, class_logits_per_layer = model(image_batch)

                # Resize masks to the resized image size
                mask_logits = F.interpolate(
                    mask_logits_per_layer[-1], image_tensor.shape[-2:], mode="bilinear"
                )

                # Multiply masks and classes to get per-pixel predictions
                per_pixel_logits = model.to_per_pixel_logits_semantic(
                    mask_logits, class_logits_per_layer[-1]
                )

                # Extract the final tensor (Shape: [Classes, H, W])
                result = per_pixel_logits[0]

            # 3. Compute all anomaly score maps at once
            anomaly_maps = compute_anomaly_maps(result)
            
            # --- Dataset Ground Truth Formatting ---
            pathGT = path.replace("images", "labels_masks")
            if "RoadObsticle21" in pathGT:
               pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT:
               pathGT = pathGT.replace("jpg", "png")
            if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
               pathGT = pathGT.replace("jpg", "png")

            if not osp.exists(pathGT):
                warnings.warn(f"GT mask not found, skipping: {pathGT}")
                continue

            mask = Image.open(pathGT)
            mask = target_transform(mask)
            ood_gts = np.array(mask)

            if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)

            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)

            if 1 not in np.unique(ood_gts):
                continue
            else:
                 per_dataset_data[dataset_name]["ood_gts"].append(ood_gts)
                 for method_name, anomaly_map in anomaly_maps.items():
                     per_dataset_data[dataset_name]["scores"][method_name].append(anomaly_map)
                 per_dataset_data[dataset_name]["image_count"] += 1
                 
            del result, anomaly_maps, ood_gts, mask
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not per_dataset_data:
        print("No valid OOD samples were found.")
        write_results_csv([], args.config, csv_path=args.output_csv)
        print(f"CSV saved to {args.output_csv}")
        return

    rows = []
    all_ood_gts = []
    all_scores = {method: [] for method in METHODS}

    for dataset_name in sorted(per_dataset_data.keys()):
        dataset_entry = per_dataset_data[dataset_name]
        if not dataset_entry["ood_gts"]:
            continue

        ood_gts = np.array(dataset_entry["ood_gts"])
        row = {
            "dataset": dataset_name,
            "images": dataset_entry["image_count"],
        }

        for method_name in METHODS:
            anomaly_scores = np.array(dataset_entry["scores"][method_name])
            prc_auc, fpr = compute_metrics(ood_gts, anomaly_scores)
            row[method_name] = {"auprc": prc_auc, "fpr95": fpr}

            all_scores[method_name].append(anomaly_scores)

        all_ood_gts.append(ood_gts)
        rows.append(row)

    if not rows:
        print("No valid OOD samples were found.")
        write_results_csv([], args.config, csv_path=args.output_csv)
        print(f"CSV saved to {args.output_csv}")
        return

    # Add one global summary row across all datasets.
    concat_ood_gts = np.concatenate(all_ood_gts, axis=0)
    all_row = {
        "dataset": "ALL",
        "images": int(sum(r["images"] for r in rows)),
    }
    for method_name in METHODS:
        concat_scores = np.concatenate(all_scores[method_name], axis=0)
        prc_auc, fpr = compute_metrics(concat_ood_gts, concat_scores)
        all_row[method_name] = {"auprc": prc_auc, "fpr95": fpr}
    rows.append(all_row)

    table = format_results_table(rows)
    print("\nEvaluation Summary (AUPRC/FPR@TPR95 in %)")
    print(table)

    write_results_csv(rows, args.config, csv_path=args.output_csv)
    print(f"CSV saved to {args.output_csv}")

    with open(txt_output_path, 'a') as file:
        file.write(f"Config: {args.config}\n")
        file.write("Evaluation Summary (AUPRC/FPR@TPR95 in %)\n")
        file.write(table + "\n\n")

if __name__ == '__main__':
    main()