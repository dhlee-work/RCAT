import json

import numpy as np
import torch


def load_region_sampling_config(json_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)

    region_groups = {
        group_name: [int(x) for x in ids]
        for group_name, ids in cfg["region_groups"].items()
    }
    group_weights = {
        group_name: float(weight)
        for group_name, weight in cfg["group_weights"].items()
    }
    return region_groups, group_weights


def sample_region_ids_group_size_aware(
    seg,
    region_groups,
    group_weights,
    min_regions=1,
    max_regions=None,
    patch_pixels=14 * 14,
    large_weight=1.0,
    small_weight=0.2,
    min_pixels=1,
):
    """Group- and size-aware anatomical region sampling."""
    seg_flat = (
        seg.detach().view(-1)
        if isinstance(seg, torch.Tensor)
        else torch.as_tensor(seg).view(-1)
    )

    ids, counts = torch.unique(seg_flat, return_counts=True)
    valid = ids > 0
    ids = ids[valid].long()
    counts = counts[valid].float()

    if ids.numel() == 0:
        return []

    valid = counts >= min_pixels
    ids = ids[valid]
    counts = counts[valid]

    if ids.numel() == 0:
        return []

    num_present = ids.numel()
    max_regions = num_present if max_regions is None else min(max_regions, num_present)
    min_regions = min(min_regions, max_regions)

    k = torch.randint(
        low=min_regions,
        high=max_regions + 1,
        size=(1,),
        device=ids.device,
    ).item()

    id_to_group = {
        int(region_id): group_name
        for group_name, group_ids in region_groups.items()
        for region_id in group_ids
    }

    weights = []
    for region_id, count in zip(ids.tolist(), counts.tolist()):
        group_name = id_to_group.get(int(region_id), "other")
        group_weight = group_weights.get(
            group_name,
            group_weights.get("other", 0.1),
        )
        size_weight = large_weight if count >= patch_pixels else small_weight
        weights.append(group_weight * size_weight)

    weights = torch.tensor(weights, device=ids.device, dtype=torch.float32)
    if weights.sum() <= 0:
        return []

    selected_idx = torch.multinomial(
        weights / weights.sum(),
        num_samples=k,
        replacement=False,
    )
    return ids[selected_idx].tolist()


def random_region_ids(seg):
    """Uniformly sample a non-empty subset of anatomical labels in a slice."""
    unique_ids = torch.unique(seg)
    unique_ids = unique_ids[unique_ids != 0]

    if unique_ids.numel() == 0:
        return []

    num_selected = np.random.choice(np.arange(1, len(unique_ids) + 1))
    selected = np.random.choice(
        unique_ids.cpu().numpy(),
        num_selected,
        replace=False,
    )
    return [int(x) for x in selected]


def make_region_mask_from_ids(seg, selected_region_ids):
    """Build a binary region condition from selected anatomical labels."""
    region_mask = torch.zeros_like(seg).long()

    for region_id in selected_region_ids:
        region_mask[seg == int(region_id)] = 1

    return region_mask


def apply_label_mapping(seg_array, mapping):
    """Map external segmentation labels to the RCAT anatomical label space."""
    seg_converted = seg_array.copy()
    mapping = dict(mapping)
    mapping["133"] = 0

    for old_value, new_value in mapping.items():
        seg_converted[seg_array == int(old_value)] = int(new_value)

    return seg_converted


def get_present_region_ids(seg):
    unique_ids = torch.unique(seg)
    unique_ids = unique_ids[unique_ids != 0]
    return [int(x) for x in unique_ids.cpu().numpy()]


def normalize_ct(slice_img, hu_min=-963, hu_max=1053):
    slice_img = np.clip(slice_img, hu_min, hu_max)
    slice_img = (slice_img - hu_min) / (hu_max - hu_min)
    return (slice_img * 255).astype(np.uint8)


def pad_to_square(array, pad_value=0.0):
    height, width = array.shape
    max_side = max(height, width)

    pad_h = max_side - height
    pad_w = max_side - width

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        array = np.pad(
            array,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )

    return array


def parse_qid(qid):
    filename, region_part = qid.split("__organ_")
    region_ids = [int(x) for x in region_part.split("-")]
    return filename, region_ids


def sample_region_ids_area_aware(
    seg,
    num_regions=1,
    min_pixels=16,
    area_alpha=0.5,
    uniform_mix=0.2,
    major_region_ids=None,
    major_boost=1.0,
):
    """Area-aware anatomical region sampling."""
    seg_flat = (
        seg.detach().view(-1)
        if isinstance(seg, torch.Tensor)
        else torch.as_tensor(seg).view(-1)
    )

    ids, counts = torch.unique(seg_flat, return_counts=True)
    valid = ids > 0
    ids = ids[valid].long()
    counts = counts[valid].float()

    if ids.numel() == 0:
        return []

    valid = counts >= min_pixels
    ids = ids[valid]
    counts = counts[valid]

    if ids.numel() == 0:
        return []

    area_prob = counts.pow(area_alpha)
    area_prob = area_prob / area_prob.sum()
    uniform_prob = torch.ones_like(area_prob) / area_prob.numel()
    prob = (1.0 - uniform_mix) * area_prob + uniform_mix * uniform_prob

    if major_region_ids is not None and major_boost != 1.0:
        major_region_ids = torch.as_tensor(
            major_region_ids,
            device=ids.device,
            dtype=ids.dtype,
        )
        is_major = torch.isin(ids, major_region_ids)
        prob = prob * torch.where(
            is_major,
            torch.full_like(prob, float(major_boost)),
            torch.ones_like(prob),
        )

    prob = prob / prob.sum()
    n = min(num_regions, ids.numel())
    selected_idx = torch.multinomial(prob, n, replacement=False)
    return ids[selected_idx].tolist()
