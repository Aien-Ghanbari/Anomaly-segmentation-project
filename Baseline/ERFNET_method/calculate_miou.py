import os
import os.path as osp
from argparse import ArgumentParser

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from dataset import cityscapes
from erfnet import ERFNet
from iouEval import iouEval
from transform import Relabel, ToLabel


NUM_CLASSES = 20


def resolve_path(path, base_dir):
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(base_dir, path))


def load_state_dict_flexible(model, state_dict):
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
            continue

        if name.startswith("module."):
            stripped = name[len("module."):]
            if stripped in model_state:
                model_state[stripped].copy_(param)
    return model


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
    default_datadir = osp.abspath(
        osp.join(script_dir, "..", "..", "cityscapes")
    )
    default_load_dir = osp.abspath(osp.join(script_dir, "..", "trained_models"))

    parser = ArgumentParser(description="Compute ERFNet mIoU on Cityscapes")
    parser.add_argument(
        "--datadir",
        default=default_datadir,
        help="Path to Cityscapes root containing leftImg8bit and gtFine",
    )
    parser.add_argument("--subset", default="val", choices=["train", "val"])
    parser.add_argument("--loadDir", default=default_load_dir)
    parser.add_argument("--loadWeights", default="erfnet_pretrained.pth")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model = ERFNet(NUM_CLASSES)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    data_dir = resolve_path(args.datadir, script_dir)
    load_dir = resolve_path(args.loadDir, script_dir)
    weights_path = resolve_path(args.loadWeights, load_dir)
    if not osp.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model = load_state_dict_flexible(model, state)
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

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = DataLoader(
        cityscapes(data_dir, input_transform, target_transform, subset=args.subset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    metric = iouEval(NUM_CLASSES)

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.max(1)[1].unsqueeze(1)
            metric.addBatch(preds, labels)

    mean_iou, per_class_iou = metric.getIoU()

    class_names = get_class_names()
    print("\nPer-class IoU (%):")
    for idx, class_name in enumerate(class_names):
        print(f"{class_name:14s}: {float(per_class_iou[idx] * 100.0):.2f}")

    print(f"\nMean IoU (%): {float(mean_iou * 100.0):.2f}")


if __name__ == "__main__":
    main()
