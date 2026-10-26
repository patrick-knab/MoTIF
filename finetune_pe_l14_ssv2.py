from __future__ import annotations

# Experimental partial vision-tower fine-tuning; not the paper's LoRA procedure.

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
UTILS_DIR = ROOT / "utils"
for path in (ROOT, UTILS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


class SSV2WindowDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        transform,
        *,
        target_t: int,
        seed: int,
        random_window: bool,
    ):
        self.rows = rows
        self.transform = transform
        self.target_t = int(target_t)
        if self.target_t < 1:
            raise ValueError("target_t must be positive")
        self.seed = int(seed)
        self.random_window = bool(random_window)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        """Vary deterministic training windows between epochs."""
        self.epoch = int(epoch)

    @staticmethod
    def _read_frames(path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode any frames from video: {path}")
        return frames

    def _sample_indices(self, n: int, index: int) -> list[int]:
        if n <= 1:
            return [0] * self.target_t
        if self.random_window:
            rng = random.Random(self.seed + self.epoch * len(self) + index)
            if n > self.target_t:
                start = rng.randint(0, n - self.target_t)
                return list(range(start, start + self.target_t))
        if self.target_t <= 1:
            return [n // 2]
        return [int(round(i * (n - 1) / (self.target_t - 1))) for i in range(self.target_t)]

    def __getitem__(self, index: int):
        row = self.rows[index]
        frames = self._read_frames(row["video_path"])
        indices = self._sample_indices(len(frames), index)
        tensors = []
        for frame_idx in indices:
            frame = frames[min(max(frame_idx, 0), len(frames) - 1)]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        clip = torch.stack(tensors, dim=0)
        return clip, int(row["class_idx"])


def load_backbone(backbone: str, device: torch.device):
    if backbone == "pe-l14":
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(device)
        transform = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
    elif backbone == "pe-g14":
        model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True).to(device)
        transform = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
    elif backbone == "l14":
        model, transform = clip.load("ViT-L/14", device=device, jit=False)
        model = model.float()
        tokenizer = clip.tokenize
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model, transform, tokenizer


def set_trainable_layers(model, train_last_blocks: int, backbone: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    model.logit_scale.requires_grad = True

    train_last_blocks = int(train_last_blocks)
    visual = model.visual
    if train_last_blocks > 0:
        for block in visual.transformer.resblocks[-train_last_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    for module_name in ("ln_post", "attn_pool"):
        module = getattr(visual, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
    if getattr(visual, "proj", None) is not None:
        visual.proj.requires_grad = True


def encode_class_text(
    model,
    tokenizer,
    class_names: list[str],
    device: torch.device,
    backbone: str,
) -> torch.Tensor:
    with torch.no_grad():
        if backbone.startswith("pe-"):
            tokens = tokenizer(class_names).to(device)
            text_features = model.encode_text(tokens, normalize=True)
        else:
            tokens = tokenizer(class_names).to(device)
            text_features = F.normalize(model.encode_text(tokens), dim=-1)
    return text_features.detach()


def encode_video(model, clips: torch.Tensor, backbone: str) -> torch.Tensor:
    if backbone.startswith("pe-"):
        # PE normalizes individual frames before averaging, but does not normalize
        # the resulting video vector. Re-normalize it before CLIP-style logits.
        return F.normalize(model.encode_video(clips, normalize=True).float(), dim=-1)

    batch_size, num_frames, channels, height, width = clips.shape
    frame_features = model.encode_image(
        clips.reshape(batch_size * num_frames, channels, height, width)
    )
    frame_features = F.normalize(frame_features.float(), dim=-1)
    video_features = frame_features.reshape(batch_size, num_frames, -1).mean(dim=1)
    return F.normalize(video_features, dim=-1)


def evaluate(
    model,
    loader,
    text_features,
    device: torch.device,
    backbone: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                image_features = encode_video(model, clips, backbone)
                logits = model.logit_scale.exp() * image_features @ text_features.t()
                loss = F.cross_entropy(logits.float(), labels)
            total_loss += float(loss.item()) * labels.numel()
            total += labels.numel()
            pred = logits.argmax(dim=1)
            top1 += int((pred == labels).sum().item())
            k = min(5, logits.shape[1])
            top5_idx = logits.topk(k, dim=1).indices
            top5 += int((top5_idx == labels[:, None]).any(dim=1).sum().item())
    if total == 0:
        return {"loss": math.nan, "top1": math.nan, "top5": math.nan}
    return {"loss": total_loss / total, "top1": top1 / total, "top5": top5 / total}


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: Path,
    split: str,
    num_classes: int,
) -> None:
    if not rows:
        raise RuntimeError(f"No rows in {manifest}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{manifest}:{index + 1} must contain a JSON object")
        if "video_path" not in row or "class_idx" not in row:
            raise ValueError(
                f"{manifest}:{index + 1} must contain video_path and class_idx"
            )
        video_path = Path(str(row["video_path"]))
        if not video_path.is_absolute():
            video_path = manifest.resolve().parent / video_path
        row["video_path"] = str(video_path.resolve())
        if not video_path.is_file():
            raise FileNotFoundError(
                f"{split} video at {manifest}:{index + 1} does not exist: {video_path}"
            )
        try:
            if isinstance(row["class_idx"], (bool, float)):
                raise ValueError("class_idx must be an integer")
            class_idx = int(row["class_idx"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{manifest}:{index + 1} has a non-integer class_idx: {row['class_idx']!r}"
            ) from error
        if not 0 <= class_idx < num_classes:
            raise ValueError(
                f"{manifest}:{index + 1} has class_idx={class_idx}, but class_names "
                f"contains {num_classes} entries"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental partial vision-tower fine-tuning on SSV2 manifests (not LoRA)."
    )
    parser.add_argument(
        "--backbone",
        choices=("pe-l14", "pe-g14", "l14"),
        default="pe-l14",
    )
    default_data_dir = ROOT / "pe_l14_ssv2_ft50"
    parser.add_argument("--manifest", type=Path, default=default_data_dir / "train_manifest.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=default_data_dir / "val_manifest.jsonl")
    parser.add_argument("--class-names", type=Path, default=default_data_dir / "class_names.json")
    parser.add_argument("--out-dir", type=Path, default=default_data_dir)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--target-t", type=int, default=8)
    parser.add_argument("--train-last-blocks", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 avoids pickling PE's Lambda transform on spawn-based systems.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=4096)
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "grad_accum", "target_t", "max_val_samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.num_workers < 0 or args.train_last_blocks < 0:
        parser.error("--num-workers and --train-last-blocks cannot be negative")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("--lr must be positive and --weight-decay cannot be negative")
    return args


def optimizer_step(model, optimizer, scaler, accumulated_samples: int) -> None:
    """Average summed gradients over the actual samples, including partial batches."""
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated_samples)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not class_names or not all(
        isinstance(name, str) and name.strip() for name in class_names
    ):
        raise ValueError(f"{args.class_names} must be a non-empty JSON list of class names")
    train_rows = read_jsonl(args.manifest)
    val_rows = read_jsonl(args.val_manifest, limit=args.max_val_samples)
    validate_rows(
        train_rows,
        manifest=args.manifest,
        split="Training",
        num_classes=len(class_names),
    )
    validate_rows(
        val_rows,
        manifest=args.val_manifest,
        split="Validation",
        num_classes=len(class_names),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, transform, tokenizer = load_backbone(args.backbone, device)
    if args.train_last_blocks > len(model.visual.transformer.resblocks):
        raise ValueError("--train-last-blocks exceeds the number of visual transformer blocks")
    set_trainable_layers(model, args.train_last_blocks, args.backbone)
    text_features = encode_class_text(
        model, tokenizer, class_names, device, args.backbone
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_ds = SSV2WindowDataset(
        train_rows, transform, target_t=args.target_t, seed=args.seed, random_window=True
    )
    val_ds = SSV2WindowDataset(
        val_rows, transform, target_t=args.target_t, seed=args.seed, random_window=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    best_top1 = -1.0
    metrics_path = args.out_dir / "metrics.jsonl"
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ds.set_epoch(epoch)
        running_loss = 0.0
        seen = 0
        accumulated_samples = 0
        for step, (clips, labels) in enumerate(train_loader, start=1):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                image_features = encode_video(model, clips, args.backbone)
                logits = model.logit_scale.exp() * image_features @ text_features.t()
                loss = F.cross_entropy(logits.float(), labels)
                scaled_loss = loss * labels.numel()
            scaler.scale(scaled_loss).backward()
            accumulated_samples += labels.numel()
            if step % args.grad_accum == 0:
                optimizer_step(model, optimizer, scaler, accumulated_samples)
                accumulated_samples = 0

            running_loss += float(loss.item()) * labels.numel()
            seen += labels.numel()
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if accumulated_samples:
            optimizer_step(model, optimizer, scaler, accumulated_samples)

        val_metrics = evaluate(
            model, val_loader, text_features, device, args.backbone
        )
        train_loss = running_loss / max(1, seen)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            "[finetune] "
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']:.4f}"
        )

        latest = args.out_dir / "latest.pt"
        payload = {
            "model": model.state_dict(),
            "class_names": class_names,
            "args": serializable_args(args),
            "metrics": row,
        }
        torch.save(payload, latest)
        if val_metrics["top1"] > best_top1:
            best_top1 = float(val_metrics["top1"])
            torch.save(payload, args.out_dir / "best.pt")

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print(f"[finetune] best checkpoint: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
