import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

import torch
from PIL import Image
import lpips
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from itertools import combinations

# ============================================================
# 1. Basic helpers
# ============================================================
@lru_cache(maxsize=4096)
def _load_grayscale_crop_array_cached(
    resolved_path,
    img_size,
    resize_method,
):
    """Load and resize a grayscale crop once and retain it in memory."""
    with Image.open(resolved_path) as img:
        img = img.convert("L")
        img = img.resize(tuple(img_size), resample=resize_method)
        arr = np.asarray(img, dtype=np.float32).copy()

    arr /= 255.0
    arr.setflags(write=False)
    return arr


def load_grayscale_crop_array(
    crop_path,
    img_size=(128, 128),
    root=".",
    resize_method=Image.BICUBIC,
):
    """
    Organ crop image를 grayscale float array로 로드.

    동일한 crop과 크기는 LRU cache에서 재사용한다.

    return:
        np.ndarray [H, W], value [0, 1]
    """
    resolved_path = resolve_crop_path(crop_path, root=root)
    resolved_path = str(Path(resolved_path).resolve())

    return _load_grayscale_crop_array_cached(
        resolved_path,
        tuple(img_size),
        int(resize_method),
    )

def summarize_metrics_by_type(metric_df, ks=[1, 5, 10]):
    summary = metric_df.mean(numeric_only=True)

    rows = []

    # non-K metrics
    for metric in ["num_query_organs", "R", "P@R"]:
        if metric in summary.index:
            rows.append({
                "metric": metric,
                "K": "-",
                "value": summary[metric],
            })

    # K-based metrics
    metric_names = [
        "Hit",
        "P",
        "Recall",
        "AP",
        "QualityPositiveN",
        "RelSizeSim",
        "RelLocSim",
        "RelLocDist",
        "BBoxIoU",
        "LPIPS",
        "SSIM",
        "MAE",
        "Wasserstein",
    ]

    for metric in metric_names:
        for k in ks:
            col = f"{metric}@{k}"
            if col in summary.index:
                rows.append({
                    "metric": metric,
                    "K": k,
                    "value": summary[col],
                })

    return pd.DataFrame(rows)

def normalize_query_organs(query_idx):
    """
    query_idx:
        int, str, list[int], list[str] 모두 대응

    return:
        set[str]
    """
    if isinstance(query_idx, (list, tuple, np.ndarray)):
        return set(map(str, query_idx))
    else:
        return {str(query_idx)}


def get_query_organ_names(query_organ_ids, query_index):
    id_map = query_index["meta"]["id_maps"]["totalseg_id_to_name"]

    organ_names = []
    for oid in sorted(query_organ_ids, key=lambda x: int(x)):
        if str(oid) in id_map:
            organ_names.append(id_map[str(oid)])
        elif int(oid) in id_map:
            organ_names.append(id_map[int(oid)])
        else:
            organ_names.append(f"organ_{oid}")

    return organ_names


def get_db_organ_info(database_index, slicename, organ_id):
    """
    retrieved slice 안에서 해당 organ_id의 organ_info를 가져옴.
    없으면 None 반환.
    """
    organ_id = str(organ_id)

    if slicename not in database_index["slices"]:
        return None

    return database_index["slices"][slicename]["organ_info"].get(organ_id, None)


def get_excluded_filenames_for_query(
    query_slicename,
    query_index,
    database_index,
):
    """
    query와 동일한 slice를 database 후보에서 제외하기 위한 filename set 생성.

    기준:
        1. slicename이 동일한 경우
        2. image_path의 파일명이 동일한 경우
    """
    excluded = {str(query_slicename)}

    # query_slice_organ = query_slicename + '__organ_' + str(query_index)
    # q_info = query_index["queries"][query_slicename]
    # q_image_name = query_slicename + '.png' # Path(q_info["image_path"]).name

    for db_slicename, db_info in database_index["slices"].items():
        # db_image_name = Path(db_info["image_path"]).name
        if len(query_slicename.split('_')) == 4:
            db_volname = db_slicename.split('_')[1]
            query_volname = query_slicename.split('_')[1]
        else:
            db_volname = db_slicename.split('_')[0]
            query_volname = query_slicename.split('_')[0]
        if db_volname == query_volname:
            excluded.add(str(db_slicename))
            # excluded.add(str(db_image_name))
    return excluded

def resolve_query_info(
    query_key,
    retrieval_result,
    query_index,
):
    """
    Resolve query information for both single-organ and multi-organ queries.

    Supported cases:
        1. query_key is already a full query id:
           s1272_slice_0047__organ_46-83

        2. query_key is a base slice id:
           s1272_slice_0047

        3. multi-organ query entry does not exist, but single-organ entries exist:
           s1272_slice_0047__organ_46
           s1272_slice_0047__organ_83

    In case 3, this function builds a temporary multi-organ q_info by merging
    single-organ query entries.
    """

    # --------------------------------------------------
    # Case 1. query_key directly exists
    # --------------------------------------------------
    if query_key in query_index["queries"]:
        q_info = query_index["queries"][query_key]
        query_organ_ids = set(map(str, q_info["query_organ_idx"]))
        return query_key, q_info, query_organ_ids

    # --------------------------------------------------
    # Parse query organs from retrieval_result
    # --------------------------------------------------
    query_organ_ids = normalize_query_organs(retrieval_result["query_idx"])
    query_organ_ids_sorted = sorted(query_organ_ids, key=lambda x: int(x))
    organ_suffix = "-".join(query_organ_ids_sorted)

    # query_key may be either:
    #   s1272_slice_0047
    #   s1272_slice_0047__organ_46
    #   s1272_slice_0047__organ_46-83
    if "__organ_" in query_key:
        base_key = query_key.split("__organ_")[0]
    else:
        base_key = query_key

    # --------------------------------------------------
    # Case 2. full multi-organ key exists
    # --------------------------------------------------
    candidate_key = f"{base_key}__organ_{organ_suffix}"

    if candidate_key in query_index["queries"]:
        q_info = query_index["queries"][candidate_key]
        query_organ_ids = set(map(str, q_info["query_organ_idx"]))
        return candidate_key, q_info, query_organ_ids

    # --------------------------------------------------
    # Case 3. full multi-organ key does not exist.
    # Merge single-organ query entries.
    # --------------------------------------------------
    single_keys = [
        f"{base_key}__organ_{organ_id}"
        for organ_id in query_organ_ids_sorted
    ]

    missing_keys = [
        key for key in single_keys
        if key not in query_index["queries"]
    ]

    if len(missing_keys) > 0:
        raise KeyError(
            f"Could not resolve query info. "
            f"query_key={query_key}, candidate_key={candidate_key}, "
            f"missing_single_organ_keys={missing_keys}"
        )

    first_info = query_index["queries"][single_keys[0]]

    merged_organ_info = {}
    query_organ_list = []

    for organ_id, single_key in zip(query_organ_ids_sorted, single_keys):
        single_info = query_index["queries"][single_key]
        organ_id_str = str(organ_id)

        if organ_id_str not in single_info["organ_info"]:
            raise KeyError(
                f"organ_id={organ_id_str} not found in organ_info of {single_key}"
            )

        merged_organ_info[organ_id_str] = single_info["organ_info"][organ_id_str]

        if "query_organ_list" in single_info:
            # single-organ query라면 보통 길이 1
            query_organ_list.extend(single_info["query_organ_list"])
        else:
            query_organ_list.append(f"organ_{organ_id_str}")

    # 중복 제거, 순서 유지
    query_organ_list = list(dict.fromkeys(query_organ_list))

    q_info = {
        "filename": first_info["filename"],
        "image_path": first_info["image_path"],
        "seg_type": first_info.get("seg_type", None),
        "seg_paths": first_info.get("seg_paths", None),

        "query_organ_idx": [int(x) for x in query_organ_ids_sorted],
        "query_organ_list": query_organ_list,

        "available_organ_idx": first_info.get("available_organ_idx", []),
        "available_organ_list": first_info.get("available_organ_list", []),

        "organ_info": merged_organ_info,
    }

    return candidate_key, q_info, set(query_organ_ids_sorted)
# ============================================================
# 2. Relevance metrics
#    P@K, Recall@K, AP@K, P@R
# ============================================================

def is_relevant_slice(query_organ_ids, slicename, database_index):
    """
    A retrieved slice is relevant only if it contains all queried organs.
    This supports both single-organ and multi-organ queries.
    """
    db_organ_info = database_index["slices"][slicename]["organ_info"]
    db_organ_ids = set(map(str, db_organ_info.keys()))

    return query_organ_ids.issubset(db_organ_ids)


def count_relevant_in_database(
    query_organ_ids,
    database_index,
    exclude_filenames=None,
):
    """
    R 계산:
        database 전체에서 query organ set을 모두 포함하는 slice 수.

    exclude_filenames:
        평가 후보에서 제외할 slice name.
        예: query와 동일한 slice.
    """
    if exclude_filenames is None:
        exclude_filenames = set()
    else:
        exclude_filenames = set(map(str, exclude_filenames))

    count = 0

    for slicename in database_index["slices"].keys():
        if str(slicename) in exclude_filenames:
            continue

        if is_relevant_slice(query_organ_ids, slicename, database_index):
            count += 1

    return count

def get_hits(
    retrieval_result,
    query_organ_ids,
    database_index,
    k=None,
    exclude_filenames=None,
):
    """
    top-k ranking에 대한 hit vector 생성.
    hit = 1 if retrieved slice contains all query organs else 0.
    """
    if exclude_filenames is None:
        exclude_filenames = set()
    else:
        exclude_filenames = set(map(str, exclude_filenames))

    top_filenames = retrieval_result["topk_filenames"]

    # 동일 slice가 혹시 들어와 있어도 제거
    top_filenames = [
        name for name in top_filenames
        if str(name) not in exclude_filenames
    ]

    if k is not None:
        top_filenames = top_filenames[:k]

    hits = []

    for slicename in top_filenames:
        is_hit = is_relevant_slice(
            query_organ_ids=query_organ_ids,
            slicename=slicename,
            database_index=database_index
        )
        hits.append(int(is_hit))

    return np.array(hits, dtype=np.float32)


def compute_precision_at_k_from_hits(hits, k):
    if len(hits) == 0:
        return np.nan

    k = min(k, len(hits))
    return float(np.mean(hits[:k]))

def compute_hit_at_k_from_hits(hits, k):
    """
    Hit@K / Success@K:
        top-K 안에 relevant result가 하나라도 있으면 1, 아니면 0.
    """
    if len(hits) == 0:
        return np.nan

    k = min(k, len(hits))
    return float(np.max(hits[:k]))

def compute_recall_at_k_from_hits(hits, num_relevant, k):
    if num_relevant == 0:
        return np.nan

    k = min(k, len(hits))
    return float(np.sum(hits[:k]) / num_relevant)


def compute_ap_at_k_from_hits(hits, num_relevant, k):
    """
    Truncated AP@K:
        AP@K = sum_i Precision@i * rel_i / min(R, K)
    """
    if num_relevant == 0:
        return np.nan

    k = min(k, len(hits))
    hits_k = hits[:k]

    if len(hits_k) == 0:
        return np.nan

    precisions = []
    num_hits = 0

    for i, rel in enumerate(hits_k, start=1):
        if rel == 1:
            num_hits += 1
            precisions.append(num_hits / i)

    denom = min(num_relevant, k)

    if denom == 0:
        return np.nan

    return float(np.sum(precisions) / denom)


def compute_r_precision_from_hits(hits, num_relevant):
    """
    P@R = R-Precision

    주의:
        topk_filenames 길이가 R보다 작으면 정확한 P@R 계산 불가.
        이 경우 np.nan 반환.
    """
    if num_relevant == 0:
        return np.nan

    if len(hits) < num_relevant:
        return np.nan

    return float(np.mean(hits[:num_relevant]))


# ============================================================
# 3. Qualitative metrics
#    Relative size, relative location, BBox IoU, LPIPS
# ============================================================

def compute_ssim_crop_similarity(
    q_organ_info,
    db_organ_info,
    img_size=(128, 128),
    root=".",
):
    """
    Organ crop 간 SSIM 계산.
    높을수록 유사함.
    """
    q_img = load_grayscale_crop_array(
        q_organ_info["crop_path"],
        img_size=img_size,
        root=root,
    )
    db_img = load_grayscale_crop_array(
        db_organ_info["crop_path"],
        img_size=img_size,
        root=root,
    )

    return float(ssim(q_img, db_img, data_range=1.0))

def compute_mae_crop_distance(
    q_organ_info,
    db_organ_info,
    img_size=(128, 128),
    root=".",
):
    """
    Organ crop 간 MAE 계산.
    낮을수록 유사함.
    """
    q_img = load_grayscale_crop_array(
        q_organ_info["crop_path"],
        img_size=img_size,
        root=root,
    )
    db_img = load_grayscale_crop_array(
        db_organ_info["crop_path"],
        img_size=img_size,
        root=root,
    )

    return float(np.mean(np.abs(q_img - db_img)))

def compute_wasserstein_crop_distance(
    q_organ_info,
    db_organ_info,
    img_size=(128, 128),
    root=".",
    foreground_only=True,
    threshold=0.0,
):
    """
    Organ crop intensity distribution 간 1D Wasserstein distance.
    낮을수록 유사함.

    foreground_only=True:
        crop에서 0 배경을 제외하고 organ foreground intensity만 비교.
    """
    q_img = load_grayscale_crop_array(
        q_organ_info["crop_path"],
        img_size=img_size,
        root=root,
    )
    db_img = load_grayscale_crop_array(
        db_organ_info["crop_path"],
        img_size=img_size,
        root=root,
    )

    if foreground_only:
        q_vals = q_img[q_img > threshold]
        db_vals = db_img[db_img > threshold]
    else:
        q_vals = q_img.reshape(-1)
        db_vals = db_img.reshape(-1)

    if len(q_vals) == 0 or len(db_vals) == 0:
        return np.nan

    return float(wasserstein_distance(q_vals, db_vals))

def compute_relative_size_similarity(
    q_organ_info,
    db_organ_info,
    area_key="foreground_relative_area",
    eps=1e-8
):
    """
    상대적 크기 유사도.
    1에 가까울수록 query organ과 retrieved organ의 상대적 크기가 유사함.
    """
    q_area = float(q_organ_info[area_key])
    db_area = float(db_organ_info[area_key])

    log_ratio = abs(math.log((db_area + eps) / (q_area + eps)))
    sim = math.exp(-log_ratio)

    return float(sim)


def compute_relative_location_similarity(
    q_organ_info,
    db_organ_info,
    centroid_key="foreground_relative_centroid"
):
    """
    상대적 위치 유사도.
    1에 가까울수록 query organ과 retrieved organ의 상대적 위치가 유사함.

    return:
        location_sim, location_dist
    """
    q_c = np.array(q_organ_info[centroid_key], dtype=np.float32)
    db_c = np.array(db_organ_info[centroid_key], dtype=np.float32)

    dist = np.linalg.norm(q_c - db_c)

    max_dist = math.sqrt(2)
    sim = 1.0 - (dist / max_dist)

    return float(sim), float(dist)


def compute_bbox_iou(box1, box2, eps=1e-8):
    """
    box format:
        [x1, y1, x2, y2]

    return:
        IoU in [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = map(float, box1)
    x2_min, y2_min, x2_max, y2_max = map(float, box2)

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union = area1 + area2 - inter_area

    return float(inter_area / (union + eps))


@lru_cache(maxsize=4096)
def _load_crop_mask_array_cached(
    resolved_path,
    img_size,
    threshold,
):
    """Load a crop mask once and retain the boolean array in memory."""
    with Image.open(resolved_path) as img:
        img = img.convert("L")
        img = img.resize(tuple(img_size), resample=Image.NEAREST)
        arr = np.asarray(img).copy()

    mask = arr > threshold
    mask.setflags(write=False)
    return mask


def load_crop_mask_from_path(
    crop_path,
    img_size=(128, 128),
    root=".",
    threshold=0,
):
    """
    LPIPS에 사용하는 masked organ crop image에서 binary mask를 생성.

    동일한 crop과 크기는 LRU cache에서 재사용한다.
    """
    resolved_path = resolve_crop_path(crop_path, root=root)
    resolved_path = str(Path(resolved_path).resolve())

    return _load_crop_mask_array_cached(
        resolved_path,
        tuple(img_size),
        int(threshold),
    )


def _compute_mask_iou(q_mask, db_mask, eps=1e-8):
    intersection = np.count_nonzero(np.logical_and(q_mask, db_mask))
    union = np.count_nonzero(np.logical_or(q_mask, db_mask))

    if union == 0:
        return np.nan

    return float((intersection + eps) / (union + eps))


def compute_crop_mask_iou_from_paths(
    q_crop_path,
    db_crop_path,
    img_size=(128, 128),
    root=".",
    threshold=0,
    eps=1e-8,
):
    """Compute crop-level binary-mask IoU using cached NumPy arrays."""
    q_mask = load_crop_mask_from_path(
        q_crop_path,
        img_size=img_size,
        root=root,
        threshold=threshold,
    )
    db_mask = load_crop_mask_from_path(
        db_crop_path,
        img_size=img_size,
        root=root,
        threshold=threshold,
    )
    return _compute_mask_iou(q_mask, db_mask, eps=eps)


def compute_bbox_iou_from_crop_paths(
    q_crop_path,
    db_crop_path,
    img_size=(128, 128),
    root=".",
    threshold=0,
    eps=1e-8,
):
    """
    기존 metric 이름은 BBoxIoU로 유지하되,
    실제 계산은 crop된 organ mask 간 IoU로 수행.
    """
    return compute_crop_mask_iou_from_paths(
        q_crop_path=q_crop_path,
        db_crop_path=db_crop_path,
        img_size=img_size,
        root=root,
        threshold=threshold,
        eps=eps,
    )


# ============================================================
# 4. LPIPS
# ============================================================

class LPIPSMetric:
    def __init__(self, net="alex", device=None, batch_size=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).eval().to(self.device)
        self.batch_size = int(batch_size)
        self.cache = {}

    @staticmethod
    @lru_cache(maxsize=512)
    def _load_crop_cpu_cached(resolved_path, img_size):
        """Cache a limited number of CPU LPIPS tensors."""
        with Image.open(resolved_path) as img:
            img = img.convert("RGB")
            img = img.resize(tuple(img_size), resample=Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32).copy()

        arr /= 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def load_crop_cpu(self, crop_path, img_size=(128, 128)):
        resolved_path = str(Path(crop_path).resolve())
        return self._load_crop_cpu_cached(
            resolved_path,
            tuple(img_size),
        )

    def load_crop(self, crop_path, img_size=(128, 128)):
        """Backward-compatible single-image loader."""
        return (
            self.load_crop_cpu(crop_path, img_size=img_size)
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )

    @torch.inference_mode()
    def batch(self, path_pairs, img_size=(128, 128), batch_size=None):
        """
        Compute LPIPS for many crop pairs with batched GPU forward passes.

        Cached pair scores are reused. Returned values follow the input order.
        """
        if len(path_pairs) == 0:
            return []

        batch_size = int(batch_size or self.batch_size)
        img_size = tuple(img_size)

        normalized_pairs = [
            (
                str(Path(path_1).resolve()),
                str(Path(path_2).resolve()),
            )
            for path_1, path_2 in path_pairs
        ]

        missing_pairs = []
        seen_missing = set()

        for path_1, path_2 in normalized_pairs:
            key = (path_1, path_2, img_size)
            if key not in self.cache and key not in seen_missing:
                missing_pairs.append((path_1, path_2))
                seen_missing.add(key)

        for start_idx in range(0, len(missing_pairs), batch_size):
            batch_pairs = missing_pairs[start_idx:start_idx + batch_size]

            image_1 = torch.stack([
                self.load_crop_cpu(path_1, img_size=img_size)
                for path_1, _ in batch_pairs
            ]).to(self.device, non_blocking=True)

            image_2 = torch.stack([
                self.load_crop_cpu(path_2, img_size=img_size)
                for _, path_2 in batch_pairs
            ]).to(self.device, non_blocking=True)

            distances = self.model.forward(
                image_1,
                image_2,
                normalize=True,
            )
            distances = distances.reshape(-1).detach().cpu().numpy()

            for (path_1, path_2), value in zip(batch_pairs, distances):
                self.cache[(path_1, path_2, img_size)] = float(value)

        return [
            self.cache[(path_1, path_2, img_size)]
            for path_1, path_2 in normalized_pairs
        ]

    @torch.inference_mode()
    def __call__(self, crop_path_1, crop_path_2, img_size=(128, 128)):
        return self.batch(
            [(crop_path_1, crop_path_2)],
            img_size=img_size,
            batch_size=1,
        )[0]

def resolve_crop_path(crop_path, root="."):
    """
    crop_path가 다음 형태여도 안전하게 처리:
        './results/...'
        'results/...'
    """
    p = Path(crop_path)

    if p.exists():
        return str(p)

    p2 = Path(root) / crop_path
    if p2.exists():
        return str(p2)

    crop_path_str = str(crop_path)

    if crop_path_str.startswith("./"):
        p3 = Path(crop_path_str[2:])
    else:
        p3 = Path("./" + crop_path_str)

    if p3.exists():
        return str(p3)

    raise FileNotFoundError(f"Crop image not found: {crop_path}")


def compute_lpips_crop_distance(
    q_organ_info,
    db_organ_info,
    lpips_metric,
    img_size=(128, 128),
    root="."
):
    """
    이미 crop된 organ image끼리 LPIPS 계산.
    낮을수록 유사함.
    """
    q_crop_path = resolve_crop_path(q_organ_info["crop_path"], root=root)
    db_crop_path = resolve_crop_path(db_organ_info["crop_path"], root=root)

    return lpips_metric(
        q_crop_path,
        db_crop_path,
        img_size=img_size
    )


# ============================================================
# 5. Quality metrics at K
# ============================================================
def _safe_nanmean(values):
    if len(values) == 0:
        return np.nan

    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return np.nan

    return float(np.nanmean(arr))


def compute_quality_metrics_at_ks(
    query_key,
    retrieval_result,
    query_index,
    database_index,
    ks=(1, 3, 5, 10),
    quality_config=None,
    lpips_metric=None,
    area_key="foreground_relative_area",
    centroid_key="foreground_relative_centroid",
    img_size=(128, 128),
    root=".",
    missing_policy="skip",
    positive_only=True,
    positive_topk=True,
):
    """
    Compute all requested quality metrics in one pass up to max(ks).

    The previous implementation recalculated the same crop pairs separately
    for every K. This implementation evaluates each selected slice/organ pair
    once and derives smaller-K results from prefix subsets.
    """
    if quality_config is None:
        quality_config = {
            "relative_location": True,
            "relative_size": True,
            "bbox_iou": True,
            "lpips": False,
            "ssim": True,
            "mae": True,
            "wasserstein": True,
        }

    ks = sorted({int(k) for k in ks if int(k) > 0})
    if len(ks) == 0:
        return {}

    _, q_info, query_organ_ids = resolve_query_info(
        query_key=query_key,
        retrieval_result=retrieval_result,
        query_index=query_index,
    )

    max_k = max(ks)
    all_top_filenames = retrieval_result["topk_filenames"]

    # Each item is (cutoff_rank, slicename). For positive_topk=True,
    # cutoff_rank is the rank among positive results. Otherwise it is the
    # original retrieval rank.
    selected_slices = []

    if positive_topk:
        positive_rank = 0
        for slicename in all_top_filenames:
            if is_relevant_slice(
                query_organ_ids=query_organ_ids,
                slicename=slicename,
                database_index=database_index,
            ):
                positive_rank += 1
                selected_slices.append((positive_rank, slicename))

            if positive_rank >= max_k:
                break
    else:
        for retrieval_rank, slicename in enumerate(
            all_top_filenames[:max_k],
            start=1,
        ):
            if positive_only and not is_relevant_slice(
                query_organ_ids=query_organ_ids,
                slicename=slicename,
                database_index=database_index,
            ):
                continue
            selected_slices.append((retrieval_rank, slicename))

    metric_names = []
    if quality_config.get("relative_size", False):
        metric_names.append("RelSizeSim")
    if quality_config.get("relative_location", False):
        metric_names.extend(["RelLocSim", "RelLocDist"])
    if quality_config.get("bbox_iou", False):
        metric_names.append("BBoxIoU")
    if quality_config.get("lpips", False):
        metric_names.append("LPIPS")
    if quality_config.get("ssim", False):
        metric_names.append("SSIM")
    if quality_config.get("mae", False):
        metric_names.append("MAE")
    if quality_config.get("wasserstein", False):
        metric_names.append("Wasserstein")

    # Slice-level values preserve the original aggregation:
    # mean over queried organs, followed by mean over selected slices.
    slice_records = []
    lpips_jobs = []

    for cutoff_rank, slicename in selected_slices:
        organ_values = {name: [] for name in metric_names}

        for organ_id in query_organ_ids:
            organ_id = str(organ_id)
            q_organ_info = q_info["organ_info"].get(organ_id, None)
            db_organ_info = get_db_organ_info(
                database_index=database_index,
                slicename=slicename,
                organ_id=organ_id,
            )

            if q_organ_info is None or db_organ_info is None:
                if missing_policy == "zero":
                    if "RelSizeSim" in organ_values:
                        organ_values["RelSizeSim"].append(0.0)
                    if "RelLocSim" in organ_values:
                        organ_values["RelLocSim"].append(0.0)
                    if "RelLocDist" in organ_values:
                        organ_values["RelLocDist"].append(np.sqrt(2))
                    if "BBoxIoU" in organ_values:
                        organ_values["BBoxIoU"].append(0.0)
                    if "LPIPS" in organ_values:
                        organ_values["LPIPS"].append(np.nan)
                    if "SSIM" in organ_values:
                        organ_values["SSIM"].append(0.0)
                    if "MAE" in organ_values:
                        organ_values["MAE"].append(np.nan)
                    if "Wasserstein" in organ_values:
                        organ_values["Wasserstein"].append(np.nan)
                continue

            if "RelSizeSim" in organ_values:
                organ_values["RelSizeSim"].append(
                    compute_relative_size_similarity(
                        q_organ_info=q_organ_info,
                        db_organ_info=db_organ_info,
                        area_key=area_key,
                    )
                )

            if "RelLocSim" in organ_values:
                rel_loc_sim, rel_loc_dist = compute_relative_location_similarity(
                    q_organ_info=q_organ_info,
                    db_organ_info=db_organ_info,
                    centroid_key=centroid_key,
                )
                organ_values["RelLocSim"].append(rel_loc_sim)
                organ_values["RelLocDist"].append(rel_loc_dist)

            if "BBoxIoU" in organ_values:
                organ_values["BBoxIoU"].append(
                    compute_bbox_iou_from_crop_paths(
                        q_crop_path=q_organ_info["crop_path"],
                        db_crop_path=db_organ_info["crop_path"],
                        img_size=img_size,
                        root=root,
                        threshold=0,
                    )
                )

            # SSIM, MAE, and Wasserstein share the cached image arrays.
            q_img = None
            db_img = None
            if any(
                name in organ_values
                for name in ("SSIM", "MAE", "Wasserstein")
            ):
                q_img = load_grayscale_crop_array(
                    q_organ_info["crop_path"],
                    img_size=img_size,
                    root=root,
                )
                db_img = load_grayscale_crop_array(
                    db_organ_info["crop_path"],
                    img_size=img_size,
                    root=root,
                )

            if "SSIM" in organ_values:
                organ_values["SSIM"].append(
                    float(ssim(q_img, db_img, data_range=1.0))
                )

            if "MAE" in organ_values:
                organ_values["MAE"].append(
                    float(np.mean(np.abs(q_img - db_img)))
                )

            if "Wasserstein" in organ_values:
                q_vals = q_img[q_img > 0.0]
                db_vals = db_img[db_img > 0.0]
                if len(q_vals) == 0 or len(db_vals) == 0:
                    organ_values["Wasserstein"].append(np.nan)
                else:
                    organ_values["Wasserstein"].append(
                        float(wasserstein_distance(q_vals, db_vals))
                    )

            if "LPIPS" in organ_values:
                if lpips_metric is None:
                    raise ValueError(
                        "lpips_metric is required when "
                        "quality_config['lpips'] is True."
                    )

                q_crop_path = resolve_crop_path(
                    q_organ_info["crop_path"],
                    root=root,
                )
                db_crop_path = resolve_crop_path(
                    db_organ_info["crop_path"],
                    root=root,
                )

                value_index = len(organ_values["LPIPS"])
                organ_values["LPIPS"].append(np.nan)
                lpips_jobs.append({
                    "paths": (q_crop_path, db_crop_path),
                    "target": organ_values["LPIPS"],
                    "index": value_index,
                })

        slice_records.append({
            "cutoff_rank": cutoff_rank,
            "organ_values": organ_values,
        })

    # Run one or a few batched LPIPS forward passes for all selected pairs.
    if len(lpips_jobs) > 0:
        lpips_values = lpips_metric.batch(
            [job["paths"] for job in lpips_jobs],
            img_size=img_size,
        )
        for job, value in zip(lpips_jobs, lpips_values):
            job["target"][job["index"]] = value

    for record in slice_records:
        record["slice_values"] = {
            metric_name: _safe_nanmean(values)
            for metric_name, values in record["organ_values"].items()
        }

    results = {}

    for k in ks:
        prefix_records = [
            record for record in slice_records
            if record["cutoff_rank"] <= k
        ]

        results[f"QualityPositiveN@{k}"] = len(prefix_records)

        for metric_name in metric_names:
            values = [
                record["slice_values"][metric_name]
                for record in prefix_records
                if metric_name in record["slice_values"]
            ]
            results[f"{metric_name}@{k}"] = _safe_nanmean(values)

    return results


def compute_quality_metrics_at_k(
    query_key,
    retrieval_result,
    query_index,
    database_index,
    k,
    quality_config=None,
    lpips_metric=None,
    area_key="foreground_relative_area",
    centroid_key="foreground_relative_centroid",
    img_size=(128, 128),
    root=".",
    missing_policy="skip",
    positive_only=True,
    positive_topk=True,
):
    """Backward-compatible wrapper for one K."""
    return compute_quality_metrics_at_ks(
        query_key=query_key,
        retrieval_result=retrieval_result,
        query_index=query_index,
        database_index=database_index,
        ks=[k],
        quality_config=quality_config,
        lpips_metric=lpips_metric,
        area_key=area_key,
        centroid_key=centroid_key,
        img_size=img_size,
        root=root,
        missing_policy=missing_policy,
        positive_only=positive_only,
        positive_topk=positive_topk,
    )


# ============================================================
# 6. Metrics for one query
# ============================================================

def compute_metrics_for_query(
    query_key,
    retrieval_result,
    query_index,
    database_index,
    ks=[1, 5, 10, 50, 100],
    quality_ks=None,
    quality_config=None,
    lpips_metric=None,
    area_key="foreground_relative_area",
    centroid_key="foreground_relative_centroid",
    img_size=(128, 128),
    root=".",
    missing_policy="skip",
):
    """
    하나의 query에 대해 relevance metric + quality metric 계산.

    quality_ks=None이면 기존 동작과 같이 ks 전체에서 quality metric을
    계산하되, 내부적으로는 max(K)까지 한 번만 계산한다.
    """
    query_idx = retrieval_result["query_idx"]
    query_organ_ids = normalize_query_organs(query_idx)

    exclude_filenames = retrieval_result.get(
        "excluded_filenames",
        [query_key],
    )

    num_relevant = count_relevant_in_database(
        query_organ_ids=query_organ_ids,
        database_index=database_index,
        exclude_filenames=exclude_filenames,
    )

    hits_all = get_hits(
        retrieval_result=retrieval_result,
        query_organ_ids=query_organ_ids,
        database_index=database_index,
        k=None,
        exclude_filenames=exclude_filenames,
    )

    results = {
        "R": num_relevant,
        "P@R": compute_r_precision_from_hits(hits_all, num_relevant),
    }

    for k in ks:
        results[f"Hit@{k}"] = compute_hit_at_k_from_hits(hits_all, k)
        results[f"P@{k}"] = compute_precision_at_k_from_hits(hits_all, k)
        results[f"Recall@{k}"] = compute_recall_at_k_from_hits(
            hits_all,
            num_relevant,
            k,
        )
        results[f"AP@{k}"] = compute_ap_at_k_from_hits(
            hits_all,
            num_relevant,
            k,
        )

    if quality_config is not None and any(quality_config.values()):
        effective_quality_ks = ks if quality_ks is None else quality_ks
        quality_results = compute_quality_metrics_at_ks(
            query_key=query_key,
            retrieval_result=retrieval_result,
            query_index=query_index,
            database_index=database_index,
            ks=effective_quality_ks,
            quality_config=quality_config,
            lpips_metric=lpips_metric,
            area_key=area_key,
            centroid_key=centroid_key,
            img_size=img_size,
            root=root,
            missing_policy=missing_policy,
            positive_only=True,
            positive_topk=True,
        )
        results.update(quality_results)

    return results

def compute_quality_detail_at_k(
    query_key,
    retrieval_result,
    query_index,
    database_index,
    k,
    quality_config=None,
    lpips_metric=None,
    area_key="foreground_relative_area",
    centroid_key="foreground_relative_centroid",
    img_size=(128, 128),
    root=".",
):
    resolved_query_key, q_info, query_organ_ids = resolve_query_info(
        query_key=query_key,
        retrieval_result=retrieval_result,
        query_index=query_index,
    )

    query_organ_names = get_query_organ_names(query_organ_ids, query_index)
    organ_name_map = {
        oid: name for oid, name in zip(
            sorted(query_organ_ids, key=lambda x: int(x)),
            query_organ_names
        )
    }

    all_top_filenames = retrieval_result["topk_filenames"]

    quality_filenames = []
    for slicename in all_top_filenames:
        if is_relevant_slice(
            query_organ_ids=query_organ_ids,
            slicename=slicename,
            database_index=database_index,
        ):
            quality_filenames.append(slicename)

        if len(quality_filenames) >= k:
            break

    rows = []

    for rank_pos, slicename in enumerate(quality_filenames, start=1):
        for organ_id in query_organ_ids:
            organ_id = str(organ_id)

            q_organ_info = q_info["organ_info"].get(organ_id, None)
            db_organ_info = get_db_organ_info(
                database_index=database_index,
                slicename=slicename,
                organ_id=organ_id,
            )

            if q_organ_info is None or db_organ_info is None:
                continue

            row = {
                "query_key": query_key,
                "resolved_query_key": resolved_query_key,
                "retrieved_slice": slicename,
                "positive_rank": rank_pos,
                "K": k,
                "organ_id": organ_id,
                "organ_name": organ_name_map.get(organ_id, f"organ_{organ_id}"),
                "query_area": float(q_organ_info.get(area_key, np.nan)),
                "db_area": float(db_organ_info.get(area_key, np.nan)),
            }

            if quality_config.get("relative_size", False):
                row["RelSizeSim"] = compute_relative_size_similarity(
                    q_organ_info=q_organ_info,
                    db_organ_info=db_organ_info,
                    area_key=area_key,
                )

            if quality_config.get("relative_location", False):
                rel_loc_sim, rel_loc_dist = compute_relative_location_similarity(
                    q_organ_info=q_organ_info,
                    db_organ_info=db_organ_info,
                    centroid_key=centroid_key,
                )
                row["RelLocSim"] = rel_loc_sim
                row["RelLocDist"] = rel_loc_dist

            if quality_config.get("bbox_iou", False):
                row["BBoxIoU"] = compute_bbox_iou_from_crop_paths(
                    q_crop_path=q_organ_info["crop_path"],
                    db_crop_path=db_organ_info["crop_path"],
                    img_size=img_size,
                    root=root,
                    threshold=0,
                )

            if quality_config.get("lpips", False):
                row["LPIPS"] = compute_lpips_crop_distance(
                    q_organ_info=q_organ_info,
                    db_organ_info=db_organ_info,
                    lpips_metric=lpips_metric,
                    img_size=img_size,
                    root=root,
                )

            if quality_config.get("ssim", False):
                row["SSIM"] = compute_ssim_crop_similarity(
                    q_organ_info=q_organ_info,
                    db_organ_info=db_organ_info,
                    img_size=img_size,
                    root=root,
                )

            if quality_config.get("mae", False):
                row["MAE"] = compute_mae_crop_distance(
                    q_organ_info=q_organ_info,
                    db_organ_info=db_organ_info,
                    img_size=img_size,
                    root=root,
                )

            if quality_config.get("wasserstein", False):
                row["Wasserstein"] = compute_wasserstein_crop_distance(
                    q_organ_info=q_organ_info,
                    db_organ_info=db_organ_info,
                    img_size=img_size,
                    root=root,
                    foreground_only=True,
                    threshold=0.0,
                )

            rows.append(row)

    return rows

# ============================================================
# 7. Evaluate all queries
# ============================================================
def evaluate_quality_details(
    retrieval_results,
    query_index,
    database_index,
    ks=[1, 3, 5, 10],
    quality_config=None,
    lpips_net="alex",
    lpips_batch_size=64,
    area_key="foreground_relative_area",
    centroid_key="foreground_relative_centroid",
    img_size=(128, 128),
    root=".",
):
    rows = []
    ks = sorted({int(k) for k in ks if int(k) > 0})

    if len(ks) == 0:
        return pd.DataFrame(rows)

    lpips_metric = None
    if quality_config is not None and quality_config.get("lpips", False):
        lpips_metric = LPIPSMetric(
            net=lpips_net,
            batch_size=lpips_batch_size,
        )

    max_k = max(ks)

    for query_key, retrieval_result in tqdm.tqdm(
        retrieval_results.items(),
        total=len(retrieval_results),
        desc="Evaluating organ-level quality details",
    ):
        max_k_rows = compute_quality_detail_at_k(
            query_key=query_key,
            retrieval_result=retrieval_result,
            query_index=query_index,
            database_index=database_index,
            k=max_k,
            quality_config=quality_config,
            lpips_metric=lpips_metric,
            area_key=area_key,
            centroid_key=centroid_key,
            img_size=img_size,
            root=root,
        )

        for k in ks:
            for source_row in max_k_rows:
                if source_row["positive_rank"] <= k:
                    row = source_row.copy()
                    row["K"] = k
                    rows.append(row)

    return pd.DataFrame(rows)

def evaluate_all_queries(
    retrieval_results,
    query_index,
    database_index,
    ks=[1, 3, 5, 10, 50, 100],
    quality_ks=None,
    quality_config=None,
    lpips_net="alex",
    lpips_batch_size=64,
    area_key="foreground_relative_area",
    centroid_key="foreground_relative_centroid",
    img_size=(128, 128),
    root=".",
    missing_policy="skip",
):
    """
    전체 query에 대해 retrieval relevance metric과 질적 metric을 계산.

    Args:
        ks:
            Retrieval metric K values.
        quality_ks:
            Quality metric K values. None이면 ks를 사용한다. 논문에서
            quality@10만 보고한다면 [10]으로 지정하는 것이 가장 빠르다.
    """
    rows = []

    lpips_metric = None
    if quality_config is not None and quality_config.get("lpips", False):
        lpips_metric = LPIPSMetric(
            net=lpips_net,
            batch_size=lpips_batch_size,
        )

    for query_key, retrieval_result in tqdm.tqdm(
        retrieval_results.items(),
        total=len(retrieval_results),
        desc="Evaluating queries",
    ):
        query_idx = retrieval_result["query_idx"]
        query_organ_ids = normalize_query_organs(query_idx)
        query_organ_names = get_query_organ_names(
            query_organ_ids,
            query_index,
        )

        metric_result = compute_metrics_for_query(
            query_key=query_key,
            retrieval_result=retrieval_result,
            query_index=query_index,
            database_index=database_index,
            ks=ks,
            quality_ks=quality_ks,
            quality_config=quality_config,
            lpips_metric=lpips_metric,
            area_key=area_key,
            centroid_key=centroid_key,
            img_size=img_size,
            root=root,
            missing_policy=missing_policy,
        )

        row = {
            "query_slice": query_key,
            "query_organ_ids": ",".join(
                sorted(query_organ_ids, key=lambda x: int(x))
            ),
            "query_organ_names": ",".join(query_organ_names),
            "num_query_organs": len(query_organ_ids),
        }

        row.update(metric_result)
        rows.append(row)

    return pd.DataFrame(rows)

def clear_metric_caches():
    """Release cached crop arrays/tensors between large experiments."""
    _load_grayscale_crop_array_cached.cache_clear()
    _load_crop_mask_array_cached.cache_clear()
    LPIPSMetric._load_crop_cpu_cached.cache_clear()


def summarize_metrics_by_organ(metric_df, ks=[1, 3, 5, 10]):
    """
    Organ-wise metric summary.

    Returns:
        organ_summary:
            organ별 평균 score table

        macro_summary:
            organ별 평균을 다시 평균낸 macro score
    """

    metric_cols = ["R", "P@R"]

    for k in ks:
        for metric in [
            "Hit",
            "P",
            "Recall",
            "AP",
            "QualityPositiveN",
            "RelSizeSim",
            "RelLocSim",
            "RelLocDist",
            "BBoxIoU",
            "LPIPS",
            "SSIM",
            "MAE",
            "Wasserstein",
        ]:
            col = f"{metric}@{k}"
            if col in metric_df.columns:
                metric_cols.append(col)

    metric_cols = [col for col in metric_cols if col in metric_df.columns]

    organ_summary = (
        metric_df
        .groupby(["query_organ_ids", "query_organ_names"])[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    organ_counts = (
        metric_df
        .groupby(["query_organ_ids", "query_organ_names"])
        .size()
        .reset_index(name="num_queries")
    )

    organ_summary = organ_counts.merge(
        organ_summary,
        on=["query_organ_ids", "query_organ_names"],
        how="left"
    )

    macro_summary = organ_summary[metric_cols].mean(numeric_only=True)

    return organ_summary, macro_summary


def summarize_metrics_by_region_group(
    metric_df,
    region_groups,
    ks=[1, 3, 5, 10],
    other_group_name="other",
):
    """
    Region-group-wise metric summary.

    Args:
        metric_df:
            Query-level metric dataframe.
            Must include query_organ_ids and preferably query_organ_names.

        region_groups:
            Dictionary such as:
            {
                "organ": [...],
                "vessel": [...],
                "bone": [...],
                "muscle": [...]
            }

        ks:
            K values used for metric columns.

        other_group_name:
            Group name for organ IDs not included in region_groups.

    Returns:
        organ_summary:
            Organ-wise average metric table with region_group.

        group_micro_summary:
            Group-wise average directly computed over all queries.

        group_macro_summary:
            Group-wise average computed after organ-wise averaging.
            This is recommended for paper tables.

        macro_summary:
            Overall macro average over organs.
    """

    metric_df = metric_df.copy()

    # --------------------------------------------------
    # 1. Build metric column list
    # --------------------------------------------------
    metric_cols = ["R", "P@R"]

    for k in ks:
        for metric in [
            "Hit",
            "P",
            "Recall",
            "AP",
            "QualityPositiveN",
            "RelSizeSim",
            "RelLocSim",
            "RelLocDist",
            "BBoxIoU",
            "LPIPS",
            "SSIM",
            "MAE",
            "Wasserstein",
        ]:
            col = f"{metric}@{k}"
            if col in metric_df.columns:
                metric_cols.append(col)

    metric_cols = [col for col in metric_cols if col in metric_df.columns]

    # --------------------------------------------------
    # 2. Map organ ID to region group
    # --------------------------------------------------
    id_to_group = {
        int(organ_id): group_name
        for group_name, organ_ids in region_groups.items()
        for organ_id in organ_ids
    }

    metric_df["region_group"] = (
        metric_df["query_organ_ids"]
        .astype(int)
        .map(id_to_group)
        .fillna(other_group_name)
    )

    # --------------------------------------------------
    # 3. Organ-wise summary
    # --------------------------------------------------
    group_keys = ["region_group", "query_organ_ids"]

    if "query_organ_names" in metric_df.columns:
        group_keys.append("query_organ_names")

    organ_summary = (
        metric_df
        .groupby(group_keys)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    organ_counts = (
        metric_df
        .groupby(group_keys)
        .size()
        .reset_index(name="num_queries")
    )

    organ_summary = organ_counts.merge(
        organ_summary,
        on=group_keys,
        how="left"
    )

    # --------------------------------------------------
    # 4. Group-wise micro summary
    #    Query 수가 많은 organ/group의 영향이 큼
    # --------------------------------------------------
    group_micro_summary = (
        metric_df
        .groupby("region_group")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    group_micro_counts = (
        metric_df
        .groupby("region_group")
        .agg(
            num_queries=("query_organ_ids", "size"),
            num_organs=("query_organ_ids", "nunique"),
        )
        .reset_index()
    )

    group_micro_summary = group_micro_counts.merge(
        group_micro_summary,
        on="region_group",
        how="left"
    )

    # --------------------------------------------------
    # 5. Group-wise macro summary
    #    Organ별 평균을 먼저 내고, 그 다음 group별 평균
    #    논문 결과표에는 이 방식 추천
    # --------------------------------------------------
    group_macro_summary = (
        organ_summary
        .groupby("region_group")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    group_macro_counts = (
        organ_summary
        .groupby("region_group")
        .agg(
            num_organs=("query_organ_ids", "nunique"),
            num_queries=("num_queries", "sum"),
        )
        .reset_index()
    )

    group_macro_summary = group_macro_counts.merge(
        group_macro_summary,
        on="region_group",
        how="left"
    )

    # --------------------------------------------------
    # 6. Overall macro summary over organs
    # --------------------------------------------------
    macro_summary = organ_summary[metric_cols].mean(numeric_only=True)

    return organ_summary, group_micro_summary, group_macro_summary, macro_summary