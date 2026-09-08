import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.checkpoint_utils import (
    inference_autocast,
    load_rcat_for_inference,
    resolve_checkpoint_path,
    resolve_device,
)
from src.datautils import RCATImageDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the RCAT database representation with a full-image condition."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/Model-Totalseg-RCAT.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, the newest project last.ckpt is used.",
    )
    parser.add_argument("--split", type=str, default="gallery")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def build_dataloader(dataset, config):
    kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.seed_number, workers=True)

    image_paths = sorted(
        glob.glob(os.path.join(config.root_dir, args.split, "image", "*.npy"))
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {config.root_dir}/{args.split}/image"
        )

    output_dir = Path(
        args.output_dir or f"./results/{config.project_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config, args.device)
    checkpoint_path = resolve_checkpoint_path(
        project_name=config.project_name,
        checkpoint=args.checkpoint,
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Database slices: {len(image_paths):,}")

    model, legacy_checkpoint = load_rcat_for_inference(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    if legacy_checkpoint:
        print("Loaded legacy checkpoint using RCAT state-dict name conversion.")

    dataset = RCATImageDataset(image_paths, config=config)
    dataloader = build_dataloader(dataset, config)

    token_batches = []
    asr_prob_batches = []
    slice_keys = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Building RCAT database"):
            imgs = batch["img"].to(device, non_blocking=True)

            with inference_autocast(device):
                encoded = model.encode_retrieval_features(
                    imgs=imgs,
                    region_masks=None,  # full-image condition
                    need_weights=False,
                )

            token_batches.append(
                encoded["anatomical_tokens"]
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )
            asr_prob_batches.append(
                encoded["asr_probs"]
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )

            slice_keys.extend(
                Path(filename).stem for filename in batch["filename"]
            )

    database_tokens = np.concatenate(token_batches, axis=0)
    database_asr_probs = np.concatenate(asr_prob_batches, axis=0)

    if database_tokens.shape[0] != len(slice_keys):
        raise RuntimeError("Database token/key count mismatch.")

    np.save(output_dir / "slice_tokens.npy", database_tokens)
    np.save(output_dir / "slice_asr_probs.npy", database_asr_probs)

    with open(output_dir / "slice_keys.json", "w") as f:
        json.dump(slice_keys, f, indent=2)

    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "legacy_checkpoint_name_conversion": bool(legacy_checkpoint),
        "split": args.split,
        "num_slices": len(slice_keys),
        "token_shape": list(database_tokens.shape),
        "asr_probability_shape": list(database_asr_probs.shape),
        "database_region_condition": "full_image",
        "uses_segmentation_for_database_construction": False,
        "asr_classes": {
            "0": "absent",
            "1": "present_nonoverlap",
            "2": "present_overlap",
        },
    }
    with open(output_dir / "database_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved database tokens: {output_dir / 'slice_tokens.npy'}")
    print(f"Saved ASR probabilities: {output_dir / 'slice_asr_probs.npy'}")
    print(f"Token array shape: {database_tokens.shape}")


if __name__ == "__main__":
    main()
