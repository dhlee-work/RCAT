import argparse
import json
import os
import pickle
from pathlib import Path

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
from src.datautils import RCATQueryDataset
from src.utils import (
    check_selected_query_organs,
    load_or_sample_query_set,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract RCAT query tokens and ASR selectivity weights."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/Model-Totalseg-RCAT.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--database_index",
        type=str,
        default="./results/totalseg_database_gt_index.json",
    )
    parser.add_argument(
        "--query_sample_dir",
        type=str,
        default="./results/query_sampling",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", choices=["one", "multi", "full"], default="one")
    parser.add_argument("--n_regions", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--min_area_pixels", type=int, default=1)
    parser.add_argument("--sampling_seed", type=int, default=777)
    parser.add_argument("--query_split", type=str, default="gallery")
    return parser.parse_args()


def query_output_name(args):
    if args.mode == "one":
        return (
            f"query_tokens_mode-one_nregion-1_"
            f"perregion-{args.n_samples}.pkl"
        )
    if args.mode == "multi":
        return (
            f"query_tokens_mode-multi_nregion-{args.n_regions}_"
            f"nquery-{args.n_queries}.pkl"
        )
    return f"query_tokens_mode-full_nquery-{args.n_queries}.pkl"


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

    if args.mode == "one" and args.n_regions != 1:
        raise ValueError("mode='one' requires --n_regions 1.")
    if args.mode == "multi" and args.n_regions < 2:
        raise ValueError("mode='multi' requires --n_regions >= 2.")

    with open(args.database_index, "r") as f:
        database_index = json.load(f)

    selected_queries = load_or_sample_query_set(
        gallery_index=database_index,
        save_dir=args.query_sample_dir,
        mode=args.mode,
        n_organs=args.n_regions,
        n_samples=args.n_samples,
        n_queries=args.n_queries,
        min_area_pixels=args.min_area_pixels,
        seed=args.sampling_seed,
        replace_if_needed=False,
    )

    expected_regions = None if args.mode == "full" else args.n_regions
    check_selected_query_organs(
        selected_queries,
        n_organs=expected_regions,
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

    model, legacy_checkpoint = load_rcat_for_inference(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    if legacy_checkpoint:
        print("Loaded legacy checkpoint using RCAT state-dict name conversion.")

    dataset = RCATQueryDataset(
        selected_queries=selected_queries,
        config=config,
        split=args.query_split,
    )
    dataloader = build_dataloader(dataset, config)

    query_results = {}

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Extracting RCAT query tokens"):
            imgs = batch["img"].to(device, non_blocking=True)
            region_masks = batch["mask"].to(device, non_blocking=True)
            query_keys = batch["query_key"]

            with inference_autocast(device):
                encoded = model.encode_retrieval_features(
                    imgs=imgs,
                    region_masks=region_masks,
                    need_weights=False,
                )

            tokens = (
                encoded["anatomical_tokens"]
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )
            asr_probs = (
                encoded["asr_probs"]
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )
            selectivity_weights = (
                encoded["selectivity_weights"]
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )

            for i, query_key in enumerate(query_keys):
                value = selected_queries[query_key]
                query_idx = (
                    value.get("query_idx")
                    if isinstance(value, dict)
                    else value
                )

                query_results[query_key] = {
                    "query_idx": [int(x) for x in query_idx],
                    "anatomical_tokens": tokens[i],
                    "selectivity_weights": selectivity_weights[i],
                    "asr_probs": asr_probs[i],
                }

    output_path = output_dir / query_output_name(args)
    with open(output_path, "wb") as f:
        pickle.dump(query_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "legacy_checkpoint_name_conversion": bool(legacy_checkpoint),
        "mode": args.mode,
        "n_regions": args.n_regions,
        "n_samples_per_region": args.n_samples if args.mode == "one" else None,
        "n_queries_requested": args.n_queries if args.mode != "one" else None,
        "num_queries_saved": len(query_results),
        "min_area_pixels": args.min_area_pixels,
        "sampling_seed": args.sampling_seed,
        "query_region_condition": "selected_region_mask",
        "selectivity_weight": "ASR probability of class 2 (present_overlap)",
    }
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved query representations: {output_path}")
    print(f"Queries: {len(query_results):,}")


if __name__ == "__main__":
    main()
