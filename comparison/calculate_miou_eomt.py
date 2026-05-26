import os
import os.path as osp
import sys
import importlib
import warnings
import inspect
from argparse import ArgumentParser

import torch
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import RepositoryNotFoundError
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import cityscapes
from iouEval import iouEval
from transform import Relabel, ToLabel


CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_NUM_CLASSES_IOU = 20


def resolve_path(path, base_dir):
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(base_dir, path))


def configure_import_paths(script_dir):
    project_root = osp.abspath(osp.join(script_dir, ".."))
    eomt_root = osp.join(project_root, "eomt")

    for path in (project_root, eomt_root):
        if osp.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _load_state_dict_any(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")


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


def load_eomt_model(config_path, device, img_size=(512, 1024), ckpt_path=None, hf_token=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    num_classes = _infer_num_classes_from_config(config)

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
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded EoMT model from local checkpoint: {ckpt_path}")
        return model

    if name is None:
        warnings.warn("No logger name found in config. Initializing randomly.")
        return model

    try:
        state_dict_path = hf_hub_download(
            repo_id=f"tue-mps/{name}",
            filename="pytorch_model.bin",
            token=hf_token,
        )
        is_dinov3 = "dinov3" in name
        if is_dinov3:
            model_kwargs["ckpt_path"] = state_dict_path
            model_kwargs["delta_weights"] = True

        if not is_dinov3:
            state_dict = _load_state_dict_any(state_dict_path, device)
            model.load_state_dict(state_dict, strict=False)

        print(f"Loaded EoMT model from HuggingFace repo tue-mps/{name}")
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
            "Pass a config with a valid pretrained model name."
        )


def get_class_names():
    return [
        "Road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]


def main():
    script_dir = osp.dirname(__file__)
    configure_import_paths(script_dir)

    default_datadir = osp.abspath(osp.join(script_dir, "..", "..", "cityscapes"))
    default_config = osp.abspath(
        osp.join(
            script_dir,
            "..",
            "eomt",
            "configs",
            "dinov2",
            "cityscapes",
            "semantic",
            "eomt_base_640.yaml",
        )
    )

    parser = ArgumentParser(description="Compute EoMT mIoU on Cityscapes")
    parser.add_argument(
        "--datadir",
        default=default_datadir,
        help="Path to Cityscapes root containing leftImg8bit and gtFine",
    )
    parser.add_argument("--subset", default="val", choices=["train", "val"])
    parser.add_argument("--config", default=default_config, help="Path to EoMT YAML config")
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Optional local checkpoint path (.bin/.pth/.ckpt). If provided, avoids HuggingFace download.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HuggingFace token for private/gated repositories.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    data_dir = resolve_path(args.datadir, script_dir)
    config_path = resolve_path(args.config, script_dir)
    ckpt_path = resolve_path(args.ckpt_path, script_dir) if args.ckpt_path else None

    if not osp.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not osp.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if ckpt_path is not None and not osp.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    model = load_eomt_model(
        config_path,
        device,
        img_size=(512, 1024),
        ckpt_path=ckpt_path,
        hf_token=args.hf_token,
    )
    if getattr(model, "num_classes", None) != CITYSCAPES_NUM_CLASSES:
        raise RuntimeError(
            "This script computes Cityscapes semantic mIoU and expects a 19-class Cityscapes model. "
            "For COCO-vs-Cityscapes comparison, use the dedicated comparison script."
        )
    model.eval()

    input_transform = Compose([
        Resize(512, Image.BILINEAR),
        ToTensor(),
    ])
    target_transform = Compose([
        Resize(512, Image.NEAREST),
        ToLabel(),
        Relabel(255, 19),
    ])

    loader = DataLoader(
        cityscapes(data_dir, input_transform, target_transform, subset=args.subset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    metric = iouEval(CITYSCAPES_NUM_CLASSES_IOU)

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            images_uint8 = images.mul(255.0).clamp(0, 255).to(torch.uint8)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                mask_logits_per_layer, class_logits_per_layer = model(images_uint8)
                mask_logits = F.interpolate(
                    mask_logits_per_layer[-1],
                    images.shape[-2:],
                    mode="bilinear",
                )
                per_pixel_logits = model.to_per_pixel_logits_semantic(
                    mask_logits,
                    class_logits_per_layer[-1],
                )

            preds = per_pixel_logits.max(1)[1].unsqueeze(1)
            metric.addBatch(preds, labels)

    mean_iou, per_class_iou = metric.getIoU()

    class_names = get_class_names()
    print("\nPer-class IoU (%):")
    for idx, class_name in enumerate(class_names):
        print(f"{class_name:14s}: {float(per_class_iou[idx] * 100.0):.2f}")

    print(f"\nMean IoU (%): {float(mean_iou * 100.0):.2f}")


if __name__ == "__main__":
    main()
