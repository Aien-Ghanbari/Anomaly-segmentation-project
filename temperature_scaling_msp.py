import glob
import hashlib
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

from erfnet import ERFNet
from ood_metrics import fpr_at_95_tpr


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20

# Keep original behavior for reproducibility/speed tradeoff.
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


def sanitize_input_pattern(pattern):
    return os.path.expanduser(str(pattern))


def stable_softmax(logits):
    logits_max = np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)


def compute_msp_map_from_logits(logits, temperature):
    temperature = max(float(temperature), 1e-6)
    probs = stable_softmax(logits / temperature)
    return 1.0 - np.max(probs, axis=0)


def build_gt_path(image_path):
    path_gt = image_path.replace("images", "labels_masks")
    if "RoadObsticle21" in path_gt:
        path_gt = path_gt.replace("webp", "png")
    if "fs_static" in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    if "RoadAnomaly" in path_gt and "RoadAnomaly21" not in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    return path_gt


def load_and_remap_gt(path_gt):
    mask = Image.open(path_gt)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in path_gt and "RoadAnomaly21" not in path_gt:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)

    if "Streethazard" in path_gt:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts


def cache_file_for_image(cache_dir, image_path):
    image_hash = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:12]
    stem = osp.splitext(osp.basename(image_path))[0]
    return osp.join(cache_dir, f"{stem}_{image_hash}.npz")


def load_cached_logits(cache_path):
    """Load cached logits and recover from corrupted/incomplete cache files."""
    try:
        with np.load(cache_path) as cache_data:
            if "logits" not in cache_data:
                raise KeyError("Missing 'logits' key")
            return cache_data["logits"].astype(np.float32)
    except (EOFError, ValueError, OSError, KeyError) as exc:
        print(f"Invalid cache file, rebuilding: {cache_path} ({exc})")
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None


def save_logits_cache_atomic(cache_path, logits, image_path, gt_path):
    """Write cache atomically to reduce the chance of partial .npz files."""
    tmp_cache_path = cache_path + ".tmp.npz"
    np.savez_compressed(
        tmp_cache_path,
        logits=logits.astype(np.float16),
        image_path=image_path,
        gt_path=gt_path,
    )
    os.replace(tmp_cache_path, cache_path)


def evaluate_with_temperature(logits_list, ood_gts_list, temperature):
    anomaly_score_list = [compute_msp_map_from_logits(logits, temperature) for logits in logits_list]

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    return prc_auc, fpr


def collect_input_paths(input_patterns):
    supported_exts = [".jpg", ".jpeg", ".png", ".webp"]
    matched_paths = []
    tried_patterns = []

    for input_pattern in input_patterns:
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
    return matched_paths, sorted(list(set(tried_patterns)))


def load_erfnet_model(load_dir, load_weights, load_model, device):
    modelpath = osp.join(load_dir, load_model)
    weightspath = osp.join(load_dir, load_weights)

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    def load_my_state_dict(model_obj, state_dict):
        own_state = model_obj.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name + " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model_obj

    model = load_my_state_dict(model, torch.load(weightspath, map_location=device))
    model.eval()
    print("Model and weights loaded successfully")
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default=[
            r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\Validation_Dataset\RoadAnomaly\images\*.jpg"
        ],
        nargs="+",
        help="Input image glob(s)",
    )
    parser.add_argument(
        "--loadDir",
        default=r"C:\Users\Aein\Desktop\Polito\Semester 2\Fundamentals of Artificial Intelligence, Machine and Deep Learning\Project\MaskArchitectureAnomaly_CourseProject-main\trained_models",
    )
    parser.add_argument("--loadWeights", default="erfnet_pretrained.pth")
    parser.add_argument("--loadModel", default="erfnet.py")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature_sweep", type=float, nargs="+", default=None)
    parser.add_argument("--save_best_temperature", action="store_true")

    parser.add_argument(
        "--logits_cache_dir",
        default=osp.abspath(osp.join(osp.dirname(__file__), "logits_cache_erfnet")),
        help="Folder used to store per-image logits as .npz",
    )
    parser.add_argument(
        "--results_txt",
        default=osp.abspath(osp.join(osp.dirname(__file__), "results_temperature_scaling.txt")),
    )
    parser.add_argument(
        "--results_json",
        default=osp.abspath(osp.join(osp.dirname(__file__), "temperature_sweep_summary.json")),
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print progress every N images (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if cuda_available:
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if (cuda_available and not args.cpu) else "cpu")
    print(f"Selected device: {device}")
    model = load_erfnet_model(args.loadDir, args.loadWeights, args.loadModel, device)

    os.makedirs(args.logits_cache_dir, exist_ok=True)

    matched_paths, tried_patterns = collect_input_paths(args.input)
    if len(matched_paths) == 0:
        print("No input images matched.")
        print("Tried patterns:")
        for pattern in tried_patterns:
            print("  - " + pattern)
        return

    print(f"Matched {len(matched_paths)} input image(s)")

    logits_list = []
    ood_gts_list = []

    for idx, image_path in enumerate(matched_paths, start=1):
        if idx % max(1, args.progress_every) == 0 or idx == 1 or idx == len(matched_paths):
            print(f"Processing image {idx}/{len(matched_paths)}")

        gt_path = build_gt_path(image_path)
        if not osp.exists(gt_path):
            print("GT not found, skipping: " + gt_path)
            continue

        ood_gts = load_and_remap_gt(gt_path)
        if 1 not in np.unique(ood_gts):
            continue

        cache_path = cache_file_for_image(args.logits_cache_dir, image_path)
        logits = None

        if osp.exists(cache_path):
            logits = load_cached_logits(cache_path)

        if logits is None:
            t0 = time.time()
            image_tensor = input_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).float().to(device)
            with torch.no_grad():
                result = model(image_tensor)
            logits = result.squeeze(0).detach().cpu().numpy().astype(np.float32)

            save_logits_cache_atomic(cache_path, logits, image_path, gt_path)

            del image_tensor, result
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if idx % max(1, args.progress_every) == 0 or idx == 1 or idx == len(matched_paths):
                print(f"Forward pass time: {time.time() - t0:.3f}s")

        logits_list.append(logits)
        ood_gts_list.append(ood_gts)

    if len(logits_list) == 0:
        print("No valid OOD samples found after GT filtering/remapping")
        return

    temperatures = args.temperature_sweep if args.temperature_sweep is not None else [args.temperature]
    summary_rows = []

    for temp in temperatures:
        prc_auc, fpr = evaluate_with_temperature(logits_list, ood_gts_list, temp)
        row = {
            "temperature": float(temp),
            "auprc_percent": float(prc_auc * 100.0),
            "fpr95_percent": float(fpr * 100.0),
        }
        summary_rows.append(row)
        print(
            f"T={row['temperature']:.4f} | AUPRC={row['auprc_percent']:.4f} | FPR95={row['fpr95_percent']:.4f}"
        )

    if args.temperature_sweep is not None and args.save_best_temperature:
        best = max(summary_rows, key=lambda r: r["auprc_percent"])
        print(
            f"BEST by AUPRC -> T={best['temperature']:.4f} | "
            f"AUPRC={best['auprc_percent']:.4f} | FPR95={best['fpr95_percent']:.4f}"
        )

    with open(args.results_txt, "a", encoding="utf-8") as txt_file:
        txt_file.write("Temperature scaling results\n")
        for row in summary_rows:
            txt_file.write(
                f"T={row['temperature']:.4f} AUPRC={row['auprc_percent']:.4f} FPR95={row['fpr95_percent']:.4f}\n"
            )
        if args.temperature_sweep is not None and args.save_best_temperature:
            best = max(summary_rows, key=lambda r: r["auprc_percent"])
            txt_file.write(
                f"BEST T={best['temperature']:.4f} AUPRC={best['auprc_percent']:.4f} FPR95={best['fpr95_percent']:.4f}\n"
            )
        txt_file.write("\n")

    with open(args.results_json, "w", encoding="utf-8") as json_file:
        json.dump(summary_rows, json_file, indent=2)

    print("Saved TXT results to " + args.results_txt)
    print("Saved JSON summary to " + args.results_json)


if __name__ == "__main__":
    main()
