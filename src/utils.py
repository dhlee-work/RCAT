import json
import os
import random
import time
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler


class WarmupStepLR(LRScheduler):
    """Warmup followed by step-wise decay while preserving group-specific LRs."""

    def __init__(
        self,
        optimizer,
        warmup_steps,
        step_size,
        gamma=0.1,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch

        if current_step < self.warmup_steps:
            warmup_factor = float(current_step + 1) / float(
                max(1, self.warmup_steps)
            )
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        steps_since_warmup = current_step - self.warmup_steps
        decay_factor = self.gamma ** (steps_since_warmup // self.step_size)
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class ProjModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_selected_region_mask(segs, selected_region):
    """Build a query-region mask from one or more anatomical label IDs."""
    device = segs.device
    masks = torch.zeros_like(segs, dtype=torch.bool)

    for b, region_ids in enumerate(selected_region):
        region_ids = torch.tensor(
            region_ids,
            device=device,
            dtype=segs.dtype,
        )
        masks[b] = torch.isin(segs[b], region_ids)

    return masks.float()


def parse_qid(qid):
    filename, organ_part = qid.split("__organ_")
    return filename, int(organ_part)


def get_volume_id_from_slice_id(slice_id):
    return slice_id.split("_slice_")[0]


def sample_gallery_queries(
    gallery_index,
    mode="one",
    n_organs=1,
    n_per_organ=20,
    n_queries=1000,
    min_area_pixels=16 * 16,
    seed=42,
    replace_if_needed=False,
):
    """Sample single-, multi-, or full-region retrieval queries."""
    rng = random.Random(seed)

    slice_to_organs = {}
    organ_to_slices = defaultdict(list)

    for filename, slice_info in gallery_index["slices"].items():
        organ_info = slice_info.get("organ_info", {})
        valid_organs = []

        for organ_id, info in organ_info.items():
            organ_id = int(organ_id)
            area = info.get("area")
            if area is not None and area >= min_area_pixels:
                valid_organs.append(organ_id)

        valid_organs = sorted(set(valid_organs))
        if not valid_organs:
            continue

        slice_to_organs[filename] = valid_organs
        for organ_id in valid_organs:
            organ_to_slices[organ_id].append(filename)

    organ_to_slices = {
        organ_id: sorted(set(filenames))
        for organ_id, filenames in organ_to_slices.items()
    }

    if mode == "one":
        if n_organs != 1:
            raise ValueError("mode='one' requires n_organs=1.")

        selected = {}
        for organ_id, filenames in sorted(organ_to_slices.items()):
            if not filenames:
                continue

            if len(filenames) >= n_per_organ:
                sampled_filenames = rng.sample(filenames, n_per_organ)
            elif replace_if_needed:
                sampled_filenames = rng.choices(filenames, k=n_per_organ)
            else:
                sampled_filenames = filenames

            for filename in sampled_filenames:
                selected[f"{filename}__organ_{organ_id}"] = [organ_id]

        return selected

    if mode == "multi":
        if n_organs < 2:
            raise ValueError("mode='multi' requires n_organs >= 2.")

        combo_to_slices = defaultdict(list)
        for filename, organs in slice_to_organs.items():
            if len(organs) < n_organs:
                continue
            for combo in combinations(organs, n_organs):
                combo_to_slices[combo].append(filename)

        combo_to_slices = {
            combo: sorted(set(filenames))
            for combo, filenames in combo_to_slices.items()
            if filenames
        }
        combos = sorted(combo_to_slices.keys())

        if not combos:
            raise ValueError(f"No valid {n_organs}-region combinations found.")

        selected = {}
        for query_idx in range(n_queries):
            combo = rng.choice(combos)
            filename = rng.choice(combo_to_slices[combo])
            organ_str = "-".join(map(str, combo))
            query_key = (
                f"{filename}__organ_{organ_str}__q{query_idx:05d}"
            )
            selected[query_key] = list(combo)

        return selected

    if mode == "full":
        filenames = sorted(slice_to_organs.keys())
        if not filenames:
            raise ValueError("No valid slices found.")

        if len(filenames) >= n_queries:
            sampled_filenames = rng.sample(filenames, n_queries)
        else:
            sampled_filenames = rng.choices(filenames, k=n_queries)

        return {
            f"{filename}__organ_-1__q{i:05d}": [-1]
            for i, filename in enumerate(sampled_filenames)
        }

    raise ValueError("mode must be one of: 'one', 'multi', 'full'.")


def check_selected_query_organs(selected, n_organs=None):
    organ_count = Counter()
    slice_count = Counter()

    for query_key, organ_ids in selected.items():
        filename = query_key.split("__organ_")[0]
        slice_count[filename] += 1
        for organ_id in organ_ids:
            organ_count[organ_id] += 1

    counts = list(organ_count.values())

    print("Number of query items:", len(selected))
    print("Number of unique slices:", len(slice_count))
    print("Total selected regions:", sum(len(v) for v in selected.values()))

    if selected:
        print("Max queries per slice:", max(slice_count.values()))
        print("Min queries per slice:", min(slice_count.values()))

    if n_organs is not None:
        assert all(len(v) == n_organs for v in selected.values())
        print(f"All query items have exactly {n_organs} regions.")

    print("\nRegion count summary")
    print("Number of regions:", len(organ_count))
    print("Min:", min(counts) if counts else 0)
    print("Max:", max(counts) if counts else 0)
    print("Mean:", sum(counts) / len(counts) if counts else 0)

    return organ_count, slice_count


def parse_gallery_query_key(query_key):
    if "__organ_" not in query_key:
        raise ValueError(f"Invalid query_key: {query_key}")

    base_slicename, organ_part = query_key.split("__organ_", 1)
    organ_part = organ_part.split("__q")[0]
    organ_ids = [int(x) for x in organ_part.split("-")]
    return base_slicename, organ_ids


def build_query_index_from_database_index(
    database_index,
    selected_query_organs,
):
    """Build a query index while preserving the existing database-index schema."""
    query_index = {
        "meta": database_index.get("meta", {}),
        "queries": {},
    }

    for query_key, q_value in selected_query_organs.items():
        query_key = str(query_key)
        base_slicename, organ_ids_from_key = parse_gallery_query_key(query_key)

        if isinstance(q_value, dict):
            query_organ_ids = q_value.get("query_idx", organ_ids_from_key)
        else:
            query_organ_ids = q_value

        query_organ_ids = [int(x) for x in query_organ_ids]
        db_slice_info = database_index["slices"][base_slicename]

        if query_organ_ids == [-1]:
            query_index["queries"][query_key] = {
                "filename": base_slicename,
                "image_path": db_slice_info.get("image_path"),
                "seg_type": db_slice_info.get("seg_type"),
                "seg_paths": db_slice_info.get("seg_paths"),
                "query_organ_idx": [-1],
                "query_organ_list": ["full_region"],
                "available_organ_idx": [],
                "available_organ_list": [],
                "organ_info": {},
            }
            continue

        db_organ_info = db_slice_info["organ_info"]
        selected_organ_info = {}

        for organ_id in query_organ_ids:
            organ_id_str = str(organ_id)
            if organ_id_str not in db_organ_info:
                raise KeyError(
                    f"organ_id={organ_id_str} not found in "
                    f"database_index['slices'][{base_slicename}]['organ_info']"
                )
            selected_organ_info[organ_id_str] = db_organ_info[organ_id_str]

        id_map = (
            database_index.get("meta", {})
            .get("id_maps", {})
            .get("totalseg_id_to_name", {})
        )
        query_organ_list = [
            id_map.get(str(organ_id), f"organ_{organ_id}")
            for organ_id in query_organ_ids
        ]

        query_index["queries"][query_key] = {
            "filename": base_slicename,
            "image_path": db_slice_info.get("image_path"),
            "seg_type": db_slice_info.get("seg_type"),
            "seg_paths": db_slice_info.get("seg_paths"),
            "query_organ_idx": query_organ_ids,
            "query_organ_list": query_organ_list,
            "available_organ_idx": list(map(int, db_organ_info.keys())),
            "available_organ_list": [],
            "organ_info": selected_organ_info,
        }

    return query_index


def prepare_selectivity_weights(weights, device, dtype, num_tokens=None):
    weights = torch.as_tensor(weights, device=device, dtype=dtype)

    if weights.dim() == 1:
        weights = weights[:, None]
    elif weights.dim() == 2:
        if weights.shape[0] == 1:
            weights = weights.T
        elif weights.shape[1] != 1:
            raise ValueError(f"Expected [O,1] or [1,O], got {weights.shape}")
    elif weights.dim() == 3:
        if weights.shape[0] == 1:
            weights = weights.squeeze(0)
        else:
            raise ValueError(f"Expected [1,O,1], got {weights.shape}")
    else:
        raise ValueError(f"Unexpected selectivity-weight shape: {weights.shape}")

    if num_tokens is not None and weights.shape[0] != num_tokens:
        raise ValueError(
            "Selectivity-weight token dimension mismatch: "
            f"{weights.shape[0]} vs expected {num_tokens}"
        )

    return weights


def selectivity_weighted_aggregate(anatomical_tokens, weights, eps=1e-6):
    """Aggregate anatomical tokens using ASR-derived selectivity weights."""
    weighted = anatomical_tokens * weights
    denom = weights.sum(dim=0).clamp_min(eps)

    if anatomical_tokens.dim() == 2:
        return weighted.sum(dim=0) / denom
    if anatomical_tokens.dim() == 3:
        return weighted.sum(dim=1) / denom

    raise ValueError(
        f"Unexpected anatomical-token shape: {anatomical_tokens.shape}"
    )


@torch.no_grad()
def retrieve_topk(
    query_tokens,
    database_tokens,
    selectivity_weights,
    db_filenames,
    topk=100,
    db_asr_flags=None,
    query_selectivity_threshold=0.1,
    exclude_filenames=None,
    time_check=False,
):
    """Retrieve database slices using selectivity-weighted token aggregation."""
    device = database_tokens.device
    dtype = database_tokens.dtype

    if time_check:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.time()

    query_tokens = torch.as_tensor(
        query_tokens,
        device=device,
        dtype=dtype,
    )

    num_tokens = database_tokens.shape[1]
    selectivity_weights = prepare_selectivity_weights(
        selectivity_weights,
        device=device,
        dtype=dtype,
        num_tokens=num_tokens,
    )

    if db_asr_flags is None:
        query_vec = selectivity_weighted_aggregate(
            query_tokens,
            selectivity_weights,
        )
        database_vecs = selectivity_weighted_aggregate(
            database_tokens,
            selectivity_weights,
        )

        query_vec = F.normalize(query_vec, dim=-1)
        database_vecs = F.normalize(database_vecs, dim=-1)
        similarity = database_vecs @ query_vec
    else:
        similarity = torch.full(
            (database_tokens.shape[0],),
            -1.0,
            device=device,
            dtype=dtype,
        )

        query_selected = (
            selectivity_weights.squeeze(-1) >= query_selectivity_threshold
        )
        db_valid = torch.logical_and(
            db_asr_flags.bool(),
            query_selected.unsqueeze(0),
        ).sum(-1) == query_selected.sum()

        valid_indices = torch.where(db_valid)[0]
        selected_database_tokens = database_tokens[valid_indices]

        if selected_database_tokens.shape[0] > 0:
            query_vec = selectivity_weighted_aggregate(
                query_tokens,
                selectivity_weights,
            )
            database_vecs = selectivity_weighted_aggregate(
                selected_database_tokens,
                selectivity_weights,
            )

            query_vec = F.normalize(query_vec, dim=-1)
            database_vecs = F.normalize(database_vecs, dim=-1)
            similarity[valid_indices] = database_vecs @ query_vec

    if exclude_filenames is None:
        exclude_filenames = set()
    else:
        exclude_filenames = set(map(str, exclude_filenames))

    db_filenames_str = np.array(list(map(str, db_filenames)))
    filename_valid_np = np.array(
        [name not in exclude_filenames for name in db_filenames_str],
        dtype=bool,
    )
    valid_mask = torch.from_numpy(filename_valid_np).to(device=device)

    # ASR-CF is a candidate filter, not a similarity penalty. Excluded database
    # slices are therefore removed from the ranked candidate set entirely.
    if db_asr_flags is not None:
        valid_mask = valid_mask & db_valid

    valid_mask_np = valid_mask.cpu().numpy().astype(bool)
    valid_similarity = similarity[valid_mask]
    valid_db_filenames = db_filenames[valid_mask_np]
    valid_original_indices = np.arange(len(db_filenames))[valid_mask_np]

    if valid_similarity.numel() == 0:
        raise ValueError("No valid database samples after excluding filenames.")

    topk = min(topk, valid_similarity.shape[0])
    topk_sim, topk_idx_valid = torch.topk(
        valid_similarity,
        k=topk,
        largest=True,
    )

    topk_idx_valid_np = topk_idx_valid.cpu().numpy()
    topk_filenames = valid_db_filenames[topk_idx_valid_np]
    topk_indices = valid_original_indices[topk_idx_valid_np]

    if time_check:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start_time
        return topk_filenames, topk_sim.cpu().numpy(), topk_indices, elapsed
    return topk_filenames, topk_sim.cpu().numpy(), topk_indices


def load_or_sample_query_set(
    gallery_index,
    save_dir,
    mode="one",
    n_organs=1,
    n_samples=50,
    n_queries=1000,
    min_area_pixels=16,
    seed=777,
    replace_if_needed=False,
):
    os.makedirs(save_dir, exist_ok=True)

    if mode == "one":
        sample_name = (
            f"query_sampling_mode-{mode}"
            f"_norgan-{n_organs}"
            f"_perorgan-{n_samples}"
            f"_minarea-{min_area_pixels}"
            f"_seed-{seed}.json"
        )
    else:
        sample_name = (
            f"query_sampling_mode-{mode}"
            f"_norgan-{n_organs}"
            f"_nquery-{n_queries}"
            f"_minarea-{min_area_pixels}"
            f"_seed-{seed}.json"
        )

    sample_path = os.path.join(save_dir, sample_name)

    if os.path.exists(sample_path):
        print(f"[Load] Existing query sampling: {sample_path}")
        with open(sample_path, "r") as f:
            region_queries = json.load(f)
        region_queries = {
            key: [int(x) for x in value]
            for key, value in region_queries.items()
        }

        # Older multi-region sampling used the slice/region combination as a
        # dictionary key and could silently overwrite repeated draws. Regenerate
        # such cached files when the requested query count is not preserved.
        if mode in {"multi", "full"} and len(region_queries) != n_queries:
            print(
                f"[Resample] Cached query count is {len(region_queries)}, "
                f"expected {n_queries}."
            )
        else:
            return region_queries

    print(f"[Sample] New query sampling: {sample_path}")

    kwargs = dict(
        gallery_index=gallery_index,
        mode=mode,
        min_area_pixels=min_area_pixels,
        seed=seed,
        replace_if_needed=replace_if_needed,
    )

    if mode == "one":
        region_queries = sample_gallery_queries(
            n_organs=n_organs,
            n_per_organ=n_samples,
            **kwargs,
        )
    elif mode == "multi":
        region_queries = sample_gallery_queries(
            n_organs=n_organs,
            n_queries=n_queries,
            **kwargs,
        )
    elif mode == "full":
        region_queries = sample_gallery_queries(
            n_queries=n_queries,
            **kwargs,
        )
    else:
        raise ValueError("mode must be one of: 'one', 'multi', 'full'.")

    with open(sample_path, "w") as f:
        json.dump(region_queries, f, indent=2)

    return region_queries
