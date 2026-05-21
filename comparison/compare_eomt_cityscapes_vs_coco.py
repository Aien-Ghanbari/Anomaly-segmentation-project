import csv
import importlib
import inspect
import os
import os.path as osp
import sys
import warnings
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import RepositoryNotFoundError
from PIL import Image
from torch.nn import functional as F

from transform import Colorize


CITYSCAPES_IGNORE_TRAINID = 255
CITYSCAPES_IGNORE_EVAL_ID = 19

# Shared “known class” set for fair COCO-vs-Cityscapes comparison.
# Cityscapes uses trainIds 0..18. COCO panoptic uses contiguous ids 0..132.
# The COCO contiguous ids below follow the standard COCO category-id order mapping:
# 1->0 person, 2->1 bicycle, 3->2 car, 4->3 motorcycle, 6->5 bus, 7->6 train, 8->7 truck, 10->9 traffic light.
COCO_CONTIG_TO_CITYSCAPES_TRAINID = {
    0: 11,  # person
    1: 18,  # bicycle
    2: 13,  # car
    3: 17,  # motorcycle
    5: 15,  # bus
    6: 16,  # train
    7: 14,  # truck
    9: 6,  # traffic light
}
KNOWN_CITYSCAPES_TRAINIDS = sorted(set(COCO_CONTIG_TO_CITYSCAPES_TRAINID.values()))


def resolve_path(path: str, base_dir: str) -> str:
    if path is None:
        return path
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(base_dir, path))


def configure_import_paths(script_dir: str) -> None:
    project_root = osp.abspath(osp.join(script_dir, ".."))
    eomt_root = osp.join(project_root, "eomt")

    for path in (project_root, eomt_root):
        if osp.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _load_state_dict_any(ckpt_path: str, device: torch.device) -> dict:
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")


def _factor_grid_from_num_patches(num_patches: int, *, prefer_square: bool = True):
    """Infer a (gh, gw) grid from a flattened patch sequence length.

    - If perfect square, returns square grid.
    - Otherwise returns a near-square factorization.
    """
    if num_patches <= 0:
        raise ValueError(f"Invalid num_patches={num_patches}")

    root = int(round(num_patches**0.5))
    if root * root == num_patches:
        return root, root

    # Find factor pair with minimal aspect distortion.
    best = None
    for gh in range(1, int(num_patches**0.5) + 1):
        if num_patches % gh != 0:
            continue
        gw = num_patches // gh
        # Score by closeness to square (or just minimal ratio).
        ratio = max(gw / gh, gh / gw)
        if best is None or ratio < best[0]:
            best = (ratio, gh, gw)

    if best is None:
        raise RuntimeError(f"Could not factor num_patches={num_patches}")
    _, gh, gw = best
    return (gh, gw) if prefer_square else (gh, gw)


def _grid_from_img_size_and_num_patches(img_size_hw, num_patches: int):
    """Infer (gh, gw) such that gh*gw=num_patches and corresponds to a square patch size."""
    h, w = int(img_size_hw[0]), int(img_size_hw[1])
    if num_patches <= 0:
        raise ValueError(f"Invalid num_patches={num_patches}")

    # Try all factor pairs and pick the one that implies equal patch size in H and W.
    candidates = []
    for gh in range(1, int(num_patches**0.5) + 1):
        if num_patches % gh != 0:
            continue
        gw = num_patches // gh
        if h % gh != 0 or w % gw != 0:
            continue
        ph = h // gh
        pw = w // gw
        if ph != pw:
            continue
        candidates.append((abs(gh - gw), gh, gw))

    if not candidates:
        # Fall back to a near-square factorization.
        return _factor_grid_from_num_patches(num_patches)

    _, gh, gw = sorted(candidates, key=lambda t: t[0])[0]
    return gh, gw


def _resize_pos_embed_2d(pos_embed: torch.Tensor, src_grid_hw, tgt_grid_hw) -> torch.Tensor:
    """Resize flattened 2D positional embedding [1, N, C] via bicubic interpolation."""
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"Expected pos_embed shape [1, N, C], got {tuple(pos_embed.shape)}")
    src_h, src_w = int(src_grid_hw[0]), int(src_grid_hw[1])
    tgt_h, tgt_w = int(tgt_grid_hw[0]), int(tgt_grid_hw[1])
    n, c = int(pos_embed.shape[1]), int(pos_embed.shape[2])
    if src_h * src_w != n:
        raise ValueError(f"src_grid {src_h}x{src_w} does not match N={n}")

    x = pos_embed.reshape(1, src_h, src_w, c).permute(0, 3, 1, 2)  # 1,C,H,W
    x = F.interpolate(x, size=(tgt_h, tgt_w), mode="bicubic", align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(1, tgt_h * tgt_w, c)
    return x


def _maybe_fix_pos_embed_in_state_dict(state_dict: dict, model: torch.nn.Module, img_size_hw) -> dict:
    """Fix common ViT pos_embed size mismatch by resizing checkpoint pos_embed to model shape."""
    key = "network.encoder.backbone.pos_embed"
    if key not in state_dict:
        return state_dict
    try:
        model_pos = model.state_dict().get(key)
        if model_pos is None:
            return state_dict
        ckpt_pos = state_dict[key]
        if not isinstance(ckpt_pos, torch.Tensor) or not isinstance(model_pos, torch.Tensor):
            return state_dict
        if ckpt_pos.shape == model_pos.shape:
            return state_dict

        src_n = int(ckpt_pos.shape[1])
        tgt_n = int(model_pos.shape[1])
        if int(ckpt_pos.shape[2]) != int(model_pos.shape[2]):
            return state_dict

        src_grid = _factor_grid_from_num_patches(src_n)
        tgt_grid = _grid_from_img_size_and_num_patches(img_size_hw, tgt_n)

        resized = _resize_pos_embed_2d(ckpt_pos.to(dtype=model_pos.dtype), src_grid, tgt_grid)
        state_dict[key] = resized
        print(
            f"Resized pos_embed from {src_grid[0]}x{src_grid[1]} ({src_n}) to {tgt_grid[0]}x{tgt_grid[1]} ({tgt_n})",
            flush=True,
        )
        return state_dict
    except Exception as e:
        raise RuntimeError(f"Failed to fix pos_embed mismatch automatically: {e}") from e


def _infer_num_classes_from_config(config: dict) -> int:
    data_cfg = config.get("data", {})
    init_args = data_cfg.get("init_args", {})
    if isinstance(init_args, dict) and "num_classes" in init_args:
        return int(init_args["num_classes"])

    class_path = data_cfg.get("class_path")
    if not class_path or "." not in class_path:
        raise KeyError("Config is missing data.class_path; cannot infer num_classes")

    module_name, class_name = class_path.rsplit(".", 1)
    data_cls = getattr(importlib.import_module(module_name), class_name)
    sig = inspect.signature(data_cls.__init__)
    param = sig.parameters.get("num_classes")
    if param is not None and param.default is not inspect._empty:
        return int(param.default)

    raise RuntimeError(
        f"Could not infer num_classes for data module '{class_path}'. "
        "Add data.init_args.num_classes to the YAML."
    )


def load_eomt_model(config_path: str, device: torch.device, img_size=(512, 1024), ckpt_path=None, hf_token=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    num_classes = _infer_num_classes_from_config(config)

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\\.Module` and is already saved during checkpointing.*",
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
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    if ckpt_path is not None:
        state_dict = _load_state_dict_any(ckpt_path, device)
        state_dict = _maybe_fix_pos_embed_in_state_dict(state_dict, model, img_size)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded EoMT model from local checkpoint: {ckpt_path}")
        return model

    if name is None:
        warnings.warn("No logger name found in config. Initializing randomly.")
        return model

    try:
        print(f"Fetching weights from HuggingFace: tue-mps/{name}", flush=True)
        state_dict_path = hf_hub_download(
            repo_id=f"tue-mps/{name}",
            filename="pytorch_model.bin",
            token=hf_token,
        )

        state_dict = _load_state_dict_any(state_dict_path, device)
        state_dict = _maybe_fix_pos_embed_in_state_dict(state_dict, model, img_size)
        model.load_state_dict(state_dict, strict=False)

        print(f"Loaded EoMT model from HuggingFace repo tue-mps/{name}", flush=True)
        return model
    except HfHubHTTPError as e:
        if getattr(e.response, "status_code", None) == 401:
            raise RuntimeError(
                f"Unauthorized (401) when downloading tue-mps/{name}. "
                "This model repo is likely gated/private. "
                "Provide --hf-token (a HuggingFace access token) or run `huggingface-cli login`."
            ) from e
        raise
    except RepositoryNotFoundError:
        raise RuntimeError(
            f"Pre-trained model repo not found for logger name '{name}'. "
            "Pass a config with a valid pretrained model name or provide --ckpt-* paths."
        )


def _cityscapes_pairs(datadir: str, subset: str):
    left_root = osp.join(datadir, "leftImg8bit", subset)
    gt_root = osp.join(datadir, "gtFine", subset)

    img_paths = sorted(glob(osp.join(left_root, "*", "*_leftImg8bit.png")))
    for img_path in img_paths:
        city = osp.basename(osp.dirname(img_path))
        stem = osp.basename(img_path).replace("_leftImg8bit.png", "")
        gt_label = osp.join(gt_root, city, f"{stem}_gtFine_labelTrainIds.png")
        gt_inst = osp.join(gt_root, city, f"{stem}_gtFine_instanceIds.png")
        if osp.isfile(gt_label) and osp.isfile(gt_inst):
            yield img_path, gt_label, gt_inst


def _load_resized_rgb_uint8(path: str, size_hw=(512, 1024)) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_resized_label(path: str, size_hw=(512, 1024)) -> np.ndarray:
    img = Image.open(path)
    img = img.resize((size_hw[1], size_hw[0]), Image.NEAREST)
    return np.array(img, dtype=np.int64)


def _map_coco_pred_to_cityscapes_known(pred_coco: np.ndarray) -> np.ndarray:
    out = np.full(pred_coco.shape, CITYSCAPES_IGNORE_EVAL_ID, dtype=np.int64)
    for coco_id, cs_id in COCO_CONTIG_TO_CITYSCAPES_TRAINID.items():
        out[pred_coco == coco_id] = cs_id
    return out


def _mask_cityscapes_pred_to_known(pred_cs: np.ndarray) -> np.ndarray:
    out = np.full(pred_cs.shape, CITYSCAPES_IGNORE_EVAL_ID, dtype=np.int64)
    for cs_id in KNOWN_CITYSCAPES_TRAINIDS:
        out[pred_cs == cs_id] = cs_id
    return out


def _update_inter_union(acc, pred_cs: np.ndarray, gt_cs: np.ndarray, valid_mask: np.ndarray):
    for cs_id in KNOWN_CITYSCAPES_TRAINIDS:
        gt_mask = (gt_cs == cs_id) & valid_mask
        pred_mask = (pred_cs == cs_id) & valid_mask
        inter = int(np.logical_and(gt_mask, pred_mask).sum())
        union = int(np.logical_or(gt_mask, pred_mask).sum())
        acc[cs_id][0] += inter
        acc[cs_id][1] += union


def _colorize_cityscapes(label_cs: np.ndarray) -> np.ndarray:
    label_cs = label_cs.astype(np.int64)
    label_cs = np.clip(label_cs, 0, CITYSCAPES_IGNORE_EVAL_ID)
    tensor = torch.from_numpy(label_cs).long().unsqueeze(0)
    rgb = Colorize(20)(tensor).permute(1, 2, 0).numpy()
    return rgb


def _draw_instance_borders(rgb: np.ndarray, sem: np.ndarray, inst: np.ndarray) -> np.ndarray:
    sem = sem.astype(np.int64)
    inst = inst.astype(np.int64)
    combined = sem * 100000 + inst
    border = np.zeros(sem.shape, dtype=bool)
    border[1:, :] |= combined[1:, :] != combined[:-1, :]
    border[:-1, :] |= combined[1:, :] != combined[:-1, :]
    border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
    border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
    out = rgb.copy()
    out[border] = 0
    return out


def main():
    script_dir = osp.dirname(__file__)
    configure_import_paths(script_dir)

    default_cityscapes_root = osp.abspath(osp.join(script_dir, "..", "..", "cityscapes"))
    default_city_cfg = osp.abspath(
        osp.join(
            script_dir,
            "..",
            "eomt",
            "configs",
            "dinov2",
            "cityscapes",
            "semantic",
            "eomt_large_1024.yaml",
        )
    )
    default_coco_cfg = osp.abspath(
        osp.join(
            script_dir,
            "..",
            "eomt",
            "configs",
            "dinov2",
            "coco",
            "panoptic",
            "eomt_large_640.yaml",
        )
    )

    parser = ArgumentParser(
        description="Compare EoMT Cityscapes-trained vs COCO-trained checkpoints on Cityscapes val (shared known classes)."
    )
    parser.add_argument("--datadir", default=default_cityscapes_root, help="Cityscapes root containing leftImg8bit/ and gtFine/")
    parser.add_argument("--subset", default="val", choices=["val", "train", "test"])
    parser.add_argument("--cityscapes-config", default=default_city_cfg)
    parser.add_argument("--coco-config", default=default_coco_cfg)
    parser.add_argument("--ckpt-cityscapes", default=osp.join(script_dir, '..', 'trained_models', 'cityscapes_semantic_eomt_large_1024.bin'), help="Optional local checkpoint for Cityscapes model")
    parser.add_argument("--ckpt-coco", default=osp.join(script_dir, '..', 'trained_models', 'coco_panoptic_eomt_large_640.bin'), help="Optional local checkpoint for COCO model")
    parser.add_argument("--hf-token", default=None, help="Optional HuggingFace token for gated/private repos")
    parser.add_argument("--max-images", type=int, default=20, help="Limit number of images for a quick run")
    parser.add_argument("--save-vis", type=int, default=10, help="How many qualitative samples to save")
    parser.add_argument("--outdir", default=osp.join(script_dir, "compare_outputs"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    datadir = resolve_path(args.datadir, script_dir)
    city_cfg = resolve_path(args.cityscapes_config, script_dir)
    coco_cfg = resolve_path(args.coco_config, script_dir)
    ckpt_city = resolve_path(args.ckpt_cityscapes, script_dir) if args.ckpt_cityscapes else None
    ckpt_coco = resolve_path(args.ckpt_coco, script_dir) if args.ckpt_coco else None
    outdir = resolve_path(args.outdir, script_dir)

    if not osp.isdir(datadir):
        raise FileNotFoundError(f"Cityscapes datadir not found: {datadir}")
    for cfg in (city_cfg, coco_cfg):
        if not osp.isfile(cfg):
            raise FileNotFoundError(f"Config not found: {cfg}")
    for ckpt in (ckpt_city, ckpt_coco):
        if ckpt is not None and not osp.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    os.makedirs(outdir, exist_ok=True)
    vis_dir = osp.join(outdir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    print("Loading Cityscapes model...")
    model_city = load_eomt_model(city_cfg, device, img_size=(512, 1024), ckpt_path=ckpt_city, hf_token=args.hf_token)
    print("Cityscapes model loaded.")
    print("Loading COCO model...")
    model_coco = load_eomt_model(coco_cfg, device, img_size=(512, 1024), ckpt_path=ckpt_coco, hf_token=args.hf_token)
    print("COCO model loaded.")

    acc_city = {cs_id: [0, 0] for cs_id in KNOWN_CITYSCAPES_TRAINIDS}
    acc_coco = {cs_id: [0, 0] for cs_id in KNOWN_CITYSCAPES_TRAINIDS}

    rows = []
    pairs = list(_cityscapes_pairs(datadir, args.subset))
    if not pairs:
        raise RuntimeError(f"No Cityscapes pairs found under {datadir} for subset='{args.subset}'")

    n = min(args.max_images, len(pairs))
    print(f"Evaluating {n} images")

    for idx, (img_path, gt_label_path, gt_inst_path) in enumerate(pairs[:n]):
        img_u8 = _load_resized_rgb_uint8(img_path).to(device)
        img_u8 = img_u8.unsqueeze(0)  # B=1

        gt_train = _load_resized_label(gt_label_path)
        gt_inst = _load_resized_label(gt_inst_path)

        valid_mask = np.isin(gt_train, KNOWN_CITYSCAPES_TRAINIDS)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                # Cityscapes-trained
                m_city, c_city = model_city(img_u8)
                mask_city = F.interpolate(m_city[-1], (512, 1024), mode="bilinear")
                logits_city = model_city.to_per_pixel_logits_semantic(mask_city, c_city[-1])
                pred_city = logits_city.argmax(1)[0].detach().cpu().numpy().astype(np.int64)

                # COCO-trained
                m_coco, c_coco = model_coco(img_u8)
                mask_coco = F.interpolate(m_coco[-1], (512, 1024), mode="bilinear")
                logits_coco = model_coco.to_per_pixel_logits_semantic(mask_coco, c_coco[-1])
                pred_coco = logits_coco.argmax(1)[0].detach().cpu().numpy().astype(np.int64)

        pred_city_known = _mask_cityscapes_pred_to_known(pred_city)
        pred_coco_known = _map_coco_pred_to_cityscapes_known(pred_coco)
        gt_known = _mask_cityscapes_pred_to_known(gt_train)

        _update_inter_union(acc_city, pred_city_known, gt_train, valid_mask)
        _update_inter_union(acc_coco, pred_coco_known, gt_train, valid_mask)

        row = {
            "idx": idx,
            "image": osp.basename(img_path),
        }
        rows.append(row)

        if idx < args.save_vis:
            # Panoptic-style preds with instance borders (qualitative)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    pred_pan_city = model_city.to_per_pixel_preds_panoptic(
                        mask_city,
                        c_city[-1],
                        model_city.stuff_classes,
                        model_city.mask_thresh,
                        model_city.overlap_thresh,
                    )[0].detach().cpu().numpy()
                    pred_pan_coco = model_coco.to_per_pixel_preds_panoptic(
                        mask_coco,
                        c_coco[-1],
                        model_coco.stuff_classes,
                        model_coco.mask_thresh,
                        model_coco.overlap_thresh,
                    )[0].detach().cpu().numpy()

            sem_pan_city = pred_pan_city[..., 0].astype(np.int64)
            inst_pan_city = pred_pan_city[..., 1].astype(np.int64)
            sem_pan_coco = pred_pan_coco[..., 0].astype(np.int64)
            inst_pan_coco = pred_pan_coco[..., 1].astype(np.int64)

            sem_pan_city_known = _mask_cityscapes_pred_to_known(sem_pan_city)
            sem_pan_coco_known = _map_coco_pred_to_cityscapes_known(sem_pan_coco)

            inst_pan_city = np.where(sem_pan_city_known == CITYSCAPES_IGNORE_EVAL_ID, 0, inst_pan_city)
            inst_pan_coco = np.where(sem_pan_coco_known == CITYSCAPES_IGNORE_EVAL_ID, 0, inst_pan_coco)

            base_name = osp.basename(img_path).replace("_leftImg8bit.png", "")

            # Input
            img_vis = Image.open(img_path).convert("RGB").resize((1024, 512), Image.BILINEAR)
            img_vis.save(osp.join(vis_dir, f"{base_name}_input.png"))

            # GT (semantic only on known set)
            gt_rgb = _colorize_cityscapes(gt_known)
            Image.fromarray(gt_rgb).save(osp.join(vis_dir, f"{base_name}_gt_known.png"))

            # Semantic preds (known)
            city_rgb = _colorize_cityscapes(pred_city_known)
            coco_rgb = _colorize_cityscapes(pred_coco_known)
            Image.fromarray(city_rgb).save(osp.join(vis_dir, f"{base_name}_pred_city_known.png"))
            Image.fromarray(coco_rgb).save(osp.join(vis_dir, f"{base_name}_pred_coco_known.png"))

            # Panoptic-style (semantic + instance borders)
            city_pan_rgb = _draw_instance_borders(city_rgb, sem_pan_city_known, inst_pan_city)
            coco_pan_rgb = _draw_instance_borders(coco_rgb, sem_pan_coco_known, inst_pan_coco)
            Image.fromarray(city_pan_rgb).save(osp.join(vis_dir, f"{base_name}_pan_city_known.png"))
            Image.fromarray(coco_pan_rgb).save(osp.join(vis_dir, f"{base_name}_pan_coco_known.png"))

            # GT instance borders (for reference)
            gt_inst_known = np.where(gt_known == CITYSCAPES_IGNORE_EVAL_ID, 0, gt_inst)
            gt_pan_rgb = _draw_instance_borders(gt_rgb, gt_known, gt_inst_known)
            Image.fromarray(gt_pan_rgb).save(osp.join(vis_dir, f"{base_name}_pan_gt_known.png"))

        if (idx + 1) % 5 == 0 or idx + 1 == n:
            print(f"Processed {idx + 1}/{n}")

    def finalize(acc):
        per_class = {}
        for cs_id, (inter, union) in acc.items():
            per_class[cs_id] = float(inter) / float(union) if union > 0 else float("nan")
        mean = float(np.nanmean(list(per_class.values())))
        return mean, per_class

    mean_city, iou_city = finalize(acc_city)
    mean_coco, iou_coco = finalize(acc_coco)

    summary_path = osp.join(outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Known-class Cityscapes-trainId set: " + ",".join(map(str, KNOWN_CITYSCAPES_TRAINIDS)) + "\n")
        f.write("COCO->Cityscapes mapping (coco_contig:cityscapes_trainId): " + str(COCO_CONTIG_TO_CITYSCAPES_TRAINID) + "\n\n")
        f.write(f"Cityscapes-trained EoMT mean IoU (known classes): {mean_city:.4f}\n")
        f.write(f"COCO-trained EoMT mean IoU (known classes): {mean_coco:.4f}\n\n")
        f.write("Per-class IoU (Cityscapes-trained):\n")
        for cs_id in KNOWN_CITYSCAPES_TRAINIDS:
            f.write(f"  {cs_id}: {iou_city[cs_id]:.4f}\n")
        f.write("Per-class IoU (COCO-trained):\n")
        for cs_id in KNOWN_CITYSCAPES_TRAINIDS:
            f.write(f"  {cs_id}: {iou_coco[cs_id]:.4f}\n")

    csv_path = osp.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mean_iou_known"] + [f"iou_cs_{cs_id}" for cs_id in KNOWN_CITYSCAPES_TRAINIDS])
        writer.writerow(["cityscapes"] + [f"{mean_city:.6f}"] + [f"{iou_city[cs_id]:.6f}" for cs_id in KNOWN_CITYSCAPES_TRAINIDS])
        writer.writerow(["coco"] + [f"{mean_coco:.6f}"] + [f"{iou_coco[cs_id]:.6f}" for cs_id in KNOWN_CITYSCAPES_TRAINIDS])

    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Saved qualitative samples under {vis_dir}")


if __name__ == "__main__":
    main()
