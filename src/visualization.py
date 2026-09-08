
import torch

from pathlib import Path

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import os

# ============================================================
# 1. Basic helpers
# ============================================================

def pad_to_square(
    arr,
    pad_value=0,
):
    """
    2D image/seg 또는 [H, W, C] image 모두 대응.
    H, W 기준으로 정방형 padding.
    """
    arr = np.asarray(arr)

    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2D array, but got shape={arr.shape}")

    H, W = arr.shape[:2]
    max_side = max(H, W)

    pad_h = max_side - H
    pad_w = max_side - W

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h == 0 and pad_w == 0:
        return arr

    if arr.ndim == 2:
        pad_width = (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        )
    else:
        pad_width = (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ) + tuple((0, 0) for _ in range(arr.ndim - 2))

    arr = np.pad(
        arr,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )

    return arr


def normalize_ct(slice_img, hu_min=-963, hu_max=1053):
    """
    CT HU image를 고정 window로 normalize.
    return: uint8, [0, 255]
    """
    slice_img = np.asarray(slice_img, dtype=np.float32)
    slice_img = np.clip(slice_img, hu_min, hu_max)
    slice_img = (slice_img - hu_min) / (hu_max - hu_min)
    slice_img = (slice_img * 255).astype(np.uint8)
    return slice_img


def prepare_image_for_visualization(
    img,
    is_ct=True,
    hu_min=-963,
    hu_max=1053,
    pad=True,
):
    """
    image visualization 전처리.

    npy CT 기준:
        raw HU -> normalize_ct -> [0, 1] -> pad_to_square

    png/jpg 기준:
        grayscale -> [0, 1] -> pad_to_square
    """
    img = np.asarray(img)

    # squeeze
    if img.ndim == 3:
        img = np.squeeze(img)

    # RGB 형태면 첫 채널만 사용
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img[..., 0]

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, but got shape={img.shape}")

    if is_ct:
        img = normalize_ct(img, hu_min=hu_min, hu_max=hu_max)
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

        if img.max() > img.min():
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
        else:
            img = np.zeros_like(img, dtype=np.float32)

    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)

    if pad:
        img = pad_to_square(img, pad_value=0.0)

    return img.astype(np.float32)


def prepare_seg_for_visualization(
    seg,
    pad=True,
):
    """
    segmentation visualization 전처리.

    label 값은 그대로 유지하고,
    padding 영역은 background=0으로 채움.
    """
    seg = np.asarray(seg)

    if seg.ndim == 3:
        seg = np.squeeze(seg)

    if seg.ndim != 2:
        raise ValueError(f"Expected 2D seg after squeeze, but got shape={seg.shape}")

    if pad:
        seg = pad_to_square(seg, pad_value=0)

    seg = np.rint(seg).astype(np.int16)

    return seg


def normalize_query_organs(query_idx):
    """
    query_idx:
        int, str, list[int], list[str], np.ndarray 모두 대응

    return:
        list[str]
    """
    if isinstance(query_idx, (list, tuple, np.ndarray)):
        return list(map(str, query_idx))
    else:
        return [str(query_idx)]


def resolve_query_info_for_visualization(
    query_key,
    retrieval_result,
    query_index,
):
    """
    query_key가 다음 둘 중 무엇이든 query 정보를 찾는다.

    1) base slice key:
        s0100_slice_0011

    2) full query key:
        s0100_slice_0011__organ_5-80
    """
    organ_ids = normalize_query_organs(retrieval_result["query_idx"])
    organ_suffix = "-".join(sorted(organ_ids, key=lambda x: int(x)))

    if query_key in query_index["queries"]:
        return query_key, query_index["queries"][query_key], organ_ids

    if "__organ_" in query_key:
        base_key = query_key.split("__organ_")[0]
    else:
        base_key = query_key

    candidate_key = f"{base_key}__organ_{organ_suffix}"

    if candidate_key in query_index["queries"]:
        return candidate_key, query_index["queries"][candidate_key], organ_ids

    # multi-organ entry가 없고 single-organ entries만 있는 경우
    single_keys = [
        f"{base_key}__organ_{oid}"
        for oid in sorted(organ_ids, key=lambda x: int(x))
    ]

    missing_keys = [k for k in single_keys if k not in query_index["queries"]]

    if len(missing_keys) > 0:
        raise KeyError(
            f"Could not resolve query info for visualization. "
            f"query_key={query_key}, candidate_key={candidate_key}, "
            f"missing_single_keys={missing_keys}"
        )

    first_info = query_index["queries"][single_keys[0]]

    merged_organ_info = {}
    query_organ_list = []

    for oid, skey in zip(sorted(organ_ids, key=lambda x: int(x)), single_keys):
        sinfo = query_index["queries"][skey]
        oid = str(oid)

        if oid not in sinfo["organ_info"]:
            raise KeyError(f"organ_id={oid} not found in {skey}")

        merged_organ_info[oid] = sinfo["organ_info"][oid]
        query_organ_list.extend(sinfo.get("query_organ_list", [f"organ_{oid}"]))

    q_info = {
        "filename": first_info["filename"],
        "image_path": first_info["image_path"],
        "seg_type": first_info.get("seg_type", None),
        "seg_paths": first_info.get("seg_paths", None),
        "query_organ_idx": [int(x) for x in sorted(organ_ids, key=lambda x: int(x))],
        "query_organ_list": list(dict.fromkeys(query_organ_list)),
        "available_organ_idx": first_info.get("available_organ_idx", []),
        "available_organ_list": first_info.get("available_organ_list", []),
        "organ_info": merged_organ_info,
    }

    return candidate_key, q_info, organ_ids


def resolve_path(path, root="."):
    """
    './xxx', 'xxx', 절대경로 모두 대응.
    """
    path = Path(path)

    if path.exists():
        return path

    path2 = Path(root) / path
    if path2.exists():
        return path2

    path_str = str(path)

    if path_str.startswith("./"):
        path3 = Path(path_str[2:])
    else:
        path3 = Path("./" + path_str)

    if path3.exists():
        return path3

    raise FileNotFoundError(f"File not found: {path}")


def load_image(
    image_path,
    root=".",
    hu_min=-963,
    hu_max=1053,
):
    """
    CT slice image 로드 + visualization 전처리.

    npy:
        raw CT HU로 보고 normalize_ct 적용

    png/jpg:
        이미 image로 저장된 것으로 보고 일반 [0,1] normalization
    """
    image_path = resolve_path(image_path, root=root)

    if image_path.suffix.lower() == ".npy":
        img = np.load(image_path)
        img = prepare_image_for_visualization(
            img,
            is_ct=True,
            hu_min=hu_min,
            hu_max=hu_max,
            pad=True,
        )
    else:
        img = Image.open(image_path).convert("L")
        img = np.asarray(img)
        img = prepare_image_for_visualization(
            img,
            is_ct=False,
            pad=True,
        )

    return img


def load_seg(seg_path, root="."):
    """
    segmentation mask 로드 + padding.

    label 값은 유지.
    padding 영역은 0으로 채움.
    """
    seg_path = resolve_path(seg_path, root=root)

    if seg_path.suffix.lower() == ".npy":
        seg = np.load(seg_path)
    else:
        seg = Image.open(seg_path)
        seg = np.asarray(seg)

    seg = prepare_seg_for_visualization(seg, pad=True)

    return seg


def get_slice_record(index, split, slicename):
    """
    split:
        'query' or 'database'

    query_index:
        index["queries"][slicename]

    database_index:
        index["slices"][slicename]
    """
    if split == "query":
        return index["queries"][slicename]
    elif split == "database":
        return index["slices"][slicename]
    else:
        raise ValueError(f"Unknown split: {split}")


def get_seg_path(slice_info):
    """
    seg_paths가 string인 경우와 dict인 경우 모두 대응.
    """
    seg_paths = slice_info["seg_paths"]

    if isinstance(seg_paths, str):
        return seg_paths

    if isinstance(seg_paths, dict):
        # 우선순위: gt segmentation
        for key in ["seg_gt", "gt", "totalseg", "vista3d"]:
            if key in seg_paths:
                return seg_paths[key]

        return list(seg_paths.values())[0]

    raise TypeError(f"Unexpected seg_paths type: {type(seg_paths)}")


# ============================================================
# 2. Overlay helpers
# ============================================================

def build_organ_mask(seg, organ_ids):
    """
    organ_ids에 해당하는 영역을 binary mask로 생성.
    """
    organ_ids_int = [int(x) for x in organ_ids]
    return np.isin(seg, organ_ids_int)


def overlay_mask_on_image(
    image,
    seg,
    organ_ids,
    alpha=0.45,
    colors=None,
):
    """
    image 위에 organ mask를 색칠.

    multi-organ query의 경우 organ마다 다른 색을 사용.
    """
    if colors is None:
        colors = [
            np.array([1.0, 0.0, 0.0]),  # red
            np.array([0.0, 1.0, 0.0]),  # green
            np.array([0.0, 0.4, 1.0]),  # blue
            np.array([1.0, 1.0, 0.0]),  # yellow
            np.array([1.0, 0.0, 1.0]),  # magenta
            np.array([0.0, 1.0, 1.0]),  # cyan
        ]

    image = np.asarray(image, dtype=np.float32)
    seg = np.asarray(seg)

    if image.shape[:2] != seg.shape[:2]:
        raise ValueError(
            f"Image and seg shape mismatch: "
            f"image={image.shape}, seg={seg.shape}"
        )

    # grayscale -> RGB
    if image.ndim == 2:
        out = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 3:
        out = image.copy()
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    organ_ids = list(map(str, organ_ids))

    for i, organ_id in enumerate(organ_ids):
        color = colors[i % len(colors)]
        mask = seg == int(organ_id)

        if mask.sum() == 0:
            continue

        out[mask] = (1 - alpha) * out[mask] + alpha * color

    return np.clip(out, 0, 1)


def has_all_organs(slice_info, organ_ids):
    """
    retrieved slice가 query organ들을 모두 포함하는지 확인.
    """
    if "organ_info" not in slice_info:
        return False

    db_organ_ids = set(map(str, slice_info["organ_info"].keys()))
    return set(map(str, organ_ids)).issubset(db_organ_ids)


def get_organ_names(organ_ids, query_index):
    id_map = query_index["meta"]["id_maps"]["totalseg_id_to_name"]

    names = []

    for oid in organ_ids:
        oid_str = str(oid)
        oid_int = int(oid)

        if oid_str in id_map:
            names.append(id_map[oid_str])
        elif oid_int in id_map:
            names.append(id_map[oid_int])
        else:
            names.append(f"organ_{oid_str}")

    return names


# ============================================================
# 3. Main visualization function
# ============================================================
#
def visualize_query_topk(
    query_key,
    retrieval_results,
    query_index,
    database_index,
    topk=10,
    r_color = None,
    root=".",
    save_dir="./results/visualization",
    save_name=None,
    alpha=0.45,
    dpi=300,
    show_score=True,
):
    os.makedirs(save_dir, exist_ok=True)

    retrieval_result = retrieval_results[query_key]

    resolved_query_key, q_info, organ_ids = resolve_query_info_for_visualization(
        query_key=query_key,
        retrieval_result=retrieval_result,
        query_index=query_index,
    )
    # organ_ids = ['1', '2']
    organ_names = get_organ_names(organ_ids, query_index)

    # --------------------------------------------------
    # Query image / seg
    # --------------------------------------------------
    q_img = load_image(q_info["image_path"], root=root)
    q_seg = load_seg(get_seg_path(q_info), root=root)

    # q_seg[q_seg==1] = 2

    q_overlay = overlay_mask_on_image(
        image=q_img,
        seg=q_seg,
        organ_ids=organ_ids,
        alpha=alpha,
    )

    top_filenames = retrieval_result["topk_filenames"][:topk]
    top_scores = retrieval_result.get(
        "topk_scores",
        [None] * len(top_filenames)
    )[:topk]

    n_total = 1 + len(top_filenames)

    fig, axes = plt.subplots(
        1,
        n_total,
        figsize=(3.0 * n_total, 3.2),
        dpi=dpi,
    )

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        ax.set_frame_on(False)

    # --------------------------------------------------
    # Query
    # --------------------------------------------------
    axes[0].imshow(q_overlay, interpolation="nearest")
    axes[0].set_title("Query", fontsize=10)

    # --------------------------------------------------
    # Top-k results
    # --------------------------------------------------
    for rank, (db_key, score) in enumerate(zip(top_filenames, top_scores), start=1):
        ax = axes[rank]

        db_info = database_index["slices"][db_key]

        db_img = load_image(db_info["image_path"], root=root)
        db_seg = load_seg(get_seg_path(db_info), root=root)

        db_overlay = overlay_mask_on_image(
            image=db_img,
            seg=db_seg,
            organ_ids=organ_ids,
            alpha=alpha,
            colors=r_color,
        )

        hit = has_all_organs(db_info, organ_ids)

        ax.imshow(db_overlay, interpolation="nearest")

        if show_score and score is not None:
            title = f"Top-{rank} : {float(score):.3f}"
        else:
            title = f"Top-{rank}"

        if not hit:
            title += "\nmissing"

        ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"Query organ: {', '.join(organ_names)}",
        fontsize=12,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.82,
        bottom=0.02,
        wspace=0.01,
        hspace=0.0,
    )

    if save_name is None:
        organ_tag = "_".join([str(o) for o in organ_ids])
        save_name = f"{query_key}_organs_{organ_tag}_top{topk}_onerow"

    png_path = os.path.join(save_dir, save_name + ".png")

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    return png_path




def visualize_query_topk_tmp(
    query_key,
    retrieval_results,
    query_index,
    database_index,
    topk=10,
    r_color = None,
    root=".",
    save_dir="./results/visualization",
    save_name=None,
    alpha=0.45,
    dpi=300,
    show_score=True,
):
    os.makedirs(save_dir, exist_ok=True)

    retrieval_result = retrieval_results[query_key]

    resolved_query_key, q_info, organ_ids = resolve_query_info_for_visualization(
        query_key=query_key,
        retrieval_result=retrieval_result,
        query_index=query_index,
    )
    # organ_ids = ['1', '2']
    organ_names = get_organ_names(organ_ids, query_index)

    # --------------------------------------------------
    # Query image / seg
    # --------------------------------------------------
    q_img = load_image(q_info["image_path"], root=root)
    q_seg = load_seg(get_seg_path(q_info), root=root)

    q_seg[q_seg==1] = 2

    q_overlay = overlay_mask_on_image(
        image=q_img,
        seg=q_seg,
        organ_ids=organ_ids,
        alpha=alpha,
    )

    top_filenames = retrieval_result["topk_filenames"][:topk]
    top_scores = retrieval_result.get(
        "topk_scores",
        [None] * len(top_filenames)
    )[:topk]

    n_total = 1 + len(top_filenames)

    fig, axes = plt.subplots(
        1,
        n_total,
        figsize=(3.0 * n_total, 3.2),
        dpi=dpi,
    )

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        ax.set_frame_on(False)

    # --------------------------------------------------
    # Query
    # --------------------------------------------------
    axes[0].imshow(q_overlay, interpolation="nearest")
    axes[0].set_title("Query", fontsize=10)

    # --------------------------------------------------
    # Top-k results
    # --------------------------------------------------
    for rank, (db_key, score) in enumerate(zip(top_filenames, top_scores), start=1):
        ax = axes[rank]

        db_info = database_index["slices"][db_key]

        db_img = load_image(db_info["image_path"], root=root)
        db_seg = load_seg(get_seg_path(db_info), root=root)

        db_overlay = overlay_mask_on_image(
            image=db_img,
            seg=db_seg,
            organ_ids=organ_ids,
            alpha=alpha,
            colors=r_color,
        )

        hit = has_all_organs(db_info, organ_ids)

        ax.imshow(db_overlay, interpolation="nearest")

        if show_score and score is not None:
            title = f"Top-{rank} : {float(score):.3f}"
        else:
            title = f"Top-{rank}"

        if not hit:
            title += "\nmissing"

        ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"Query organ: {', '.join(organ_names)}",
        fontsize=12,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.82,
        bottom=0.02,
        wspace=0.01,
        hspace=0.0,
    )

    if save_name is None:
        organ_tag = "_".join([str(o) for o in organ_ids])
        save_name = f"{query_key}_organs_{organ_tag}_top{topk}_onerow"

    png_path = os.path.join(save_dir, save_name + ".png")

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    return png_path

def to_numpy_organs(selected_organs):
    if isinstance(selected_organs, torch.Tensor):
        selected_organs = selected_organs.detach().cpu().numpy()
    selected_organs = np.asarray(selected_organs).astype(int).tolist()
    return selected_organs


def to_numpy_scores(topk_sim):
    if isinstance(topk_sim, torch.Tensor):
        topk_sim = topk_sim.detach().float().cpu().numpy()
    return np.asarray(topk_sim).astype(float)


def to_numpy_filenames(topk_filenames):
    topk_filenames = np.asarray(topk_filenames)
    return [str(x) for x in topk_filenames.tolist()]


def make_image_path(root_dir, slice_key):
    slice_key = os.path.splitext(str(slice_key))[0]
    return os.path.join(root_dir, "gallery", "image", slice_key + ".npy")


def make_seg_path(root_dir, slice_key):
    slice_key = os.path.splitext(str(slice_key))[0]
    return os.path.join(root_dir, "gallery", "seg_gt", slice_key + ".npy")


def load_ct_image_for_vis(path):
    img = np.load(path)

    # 2D CT
    if img.ndim == 3:
        img = img.squeeze()

    img = normalize_ct(img)
    img = pad_to_square(img)

    img = np.asarray(img, dtype=np.float32)

    # 혹시 normalize_ct 결과가 0~1이 아니면 보정
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)

    return img


def load_seg_for_vis(path):
    seg = np.load(path)

    if seg.ndim == 3:
        seg = seg.squeeze()

    seg = pad_to_square(seg)
    seg = np.rint(seg).astype(np.int16)

    return seg


def overlay_selected_organs(
    image,
    seg,
    selected_organs,
    alpha=0.45,
):
    """
    image: [H, W], float 0~1
    seg: [H, W], int
    selected_organs: list of organ ids, e.g., [81] or [81, 42, 43]
    """

    image_rgb = np.stack([image, image, image], axis=-1)

    # 최대 4개 장기용. 더 많아도 반복 사용 가능.
    colors = np.array([
        [1.00, 0.10, 0.10],  # red
        [0.10, 0.70, 1.00],  # blue
        [0.10, 0.90, 0.25],  # green
        [1.00, 0.80, 0.10],  # yellow
    ], dtype=np.float32)

    overlay = image_rgb.copy()

    for idx, organ_id in enumerate(selected_organs):
        color = colors[idx % len(colors)]
        mask = seg == organ_id

        if mask.sum() == 0:
            continue

        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    overlay = np.clip(overlay, 0, 1)

    return overlay


def has_all_selected_organs(seg, selected_organs):
    present = set(np.unique(seg).astype(int).tolist())
    return all(int(o) in present for o in selected_organs)


def get_selected_organ_presence(seg, selected_organs):
    """
    Returns
    -------
    dict
        {
            organ_id: True/False,
            ...
        }
    """
    if torch.is_tensor(seg):
        seg = seg.detach().cpu().numpy()

    selected_organs = np.asarray(selected_organs).reshape(-1)
    unique_ids = set(np.unique(seg).tolist())

    return {
        int(organ_id): int(organ_id) in unique_ids
        for organ_id in selected_organs
    }


def visualize_multi_query_topk(
    query_slide_name,
    topk_filenames,
    topk_sim,
    selected_organs,
    root_dir,
    intr_organs=None,
    query_bbox=None,
    bbox_source_shape=None,
    topk=10,
    save_dir="./results/retrieval_visualization",
    save_name=None,
    alpha=0.45,
    dpi=300,
    show_score=True,
    show_missing=True,
    highlight_intr_organs=False,
    vis=False,
):
    """
    Multi-region retrieval 결과 시각화.

    Parameters
    ----------
    query_slide_name : str
        Query slice 이름.

    topk_filenames : list or np.ndarray
        검색 결과 slice key 목록.

    topk_sim : torch.Tensor or np.ndarray
        검색 유사도 점수.

    selected_organs : torch.Tensor or list
        실제 query에 사용된 organ ID.

    root_dir : str
        데이터 root directory.

    intr_organs : torch.Tensor or list, optional
        Retrieval 결과에서 presence를 확인할 관심 organ.
        None이면 selected_organs를 사용.

    query_bbox : tuple, optional
        Query ROI bbox.
        (x1, y1, x2, y2)

    bbox_source_shape : tuple, optional
        query_bbox가 정의된 원본 spatial shape.

    topk : int
        표시할 retrieval 결과 수.

    highlight_intr_organs : bool
        False:
            Query와 retrieval 결과 모두 selected_organs만 색칠.

        True:
            Query는 selected_organs만 색칠하고,
            retrieval 결과에서는 selected_organs와 intr_organs를
            모두 색칠.

    vis : bool
        True이면 plt.show().
    """

    from matplotlib.patches import Patch, Rectangle

    os.makedirs(save_dir, exist_ok=True)

    # ==================================================
    # 0. Input 정리
    # ==================================================
    selected_organs = to_numpy_organs(selected_organs)

    if intr_organs is None:
        intr_organs = selected_organs
    else:
        intr_organs = to_numpy_organs(intr_organs)

    # --------------------------------------------------
    # Retrieval 결과에서 overlay할 organ
    #
    # selected_organs를 앞에 유지하여
    # query와 retrieval 결과에서 query organ의
    # color index가 동일하도록 함
    # --------------------------------------------------
    if highlight_intr_organs:
        overlay_organs = np.asarray(
            list(
                dict.fromkeys(
                    [int(o) for o in selected_organs]
                    + [int(o) for o in intr_organs]
                )
            ),
            dtype=int,
        )
    else:
        overlay_organs = np.asarray(
            [int(o) for o in selected_organs],
            dtype=int,
        )

    topk_filenames = to_numpy_filenames(topk_filenames)
    topk_scores = to_numpy_scores(topk_sim)

    topk_filenames = topk_filenames[:topk]
    topk_scores = topk_scores[:topk]

    # ==================================================
    # 1. Query load
    # ==================================================
    q_img_path = make_image_path(
        root_dir,
        query_slide_name,
    )

    q_seg_path = make_seg_path(
        root_dir,
        query_slide_name,
    )

    q_img = load_ct_image_for_vis(q_img_path)
    q_seg = load_seg_for_vis(q_seg_path)

    # ==================================================
    # 2. Visualization용 bbox 좌표 변환
    # ==================================================
    vis_bbox = None

    if query_bbox is not None:

        x1, y1, x2, y2 = query_bbox

        if bbox_source_shape is not None:

            src_h, src_w = bbox_source_shape
            dst_h, dst_w = q_seg.shape[-2:]

            scale_x = dst_w / src_w
            scale_y = dst_h / src_h

            vis_bbox = (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            )

        else:
            vis_bbox = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )

    # ==================================================
    # 3. Query organ overlay
    # ==================================================
    # Query에서는 실제 검색에 사용된 selected_organs만 표시
    q_overlay = overlay_selected_organs(
        image=q_img,
        seg=q_seg,
        selected_organs=selected_organs,
        alpha=alpha,
    )

    # ==================================================
    # 4. Figure
    # ==================================================
    n_cols = 1 + len(topk_filenames)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(2.6 * n_cols, 3.1),
        dpi=dpi,
    )

    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # ==================================================
    # 5. Query
    # ==================================================
    axes[0].imshow(
        q_overlay,
        interpolation="nearest",
    )

    # bbox 표시
    if vis_bbox is not None:

        x1, y1, x2, y2 = vis_bbox

        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="yellow",
            linewidth=2.0,
        )

        axes[0].add_patch(rect)

    organ_text = ", ".join(
        str(int(o))
        for o in selected_organs
    )

    axes[0].set_title(
        f"Query\n{query_slide_name}\n"
        f"Regions: {organ_text}",
        fontsize=8,
    )

    # ==================================================
    # 6. Top-k results
    # ==================================================
    for rank, (db_key, score) in enumerate(
        zip(topk_filenames, topk_scores),
        start=1,
    ):

        ax = axes[rank]

        db_img_path = make_image_path(
            root_dir,
            db_key,
        )

        db_seg_path = make_seg_path(
            root_dir,
            db_key,
        )

        db_img = load_ct_image_for_vis(
            db_img_path
        )

        db_seg = load_seg_for_vis(
            db_seg_path
        )

        # --------------------------------------------------
        # Retrieval 결과 overlay
        #
        # highlight_intr_organs=False:
        #     selected_organs만 표시
        #
        # highlight_intr_organs=True:
        #     selected_organs + intr_organs 표시
        # --------------------------------------------------
        db_overlay = overlay_selected_organs(
            image=db_img,
            seg=db_seg,
            selected_organs=overlay_organs,
            alpha=alpha,
        )

        # --------------------------------------------------
        # Presence
        # --------------------------------------------------
        organ_presence = get_selected_organ_presence(
            db_seg,
            intr_organs,
        )

        if show_missing:
            status_text = " ".join(
                f"{organ_id}{'✓' if present else '✗'}"
                for organ_id, present
                in organ_presence.items()
            )
        else:
            status_text = " ".join(
                str(organ_id)
                for organ_id, present
                in organ_presence.items()
                if present
            )

        ax.imshow(
            db_overlay,
            interpolation="nearest",
        )

        if show_score:

            if status_text:
                title = (
                    f"Top-{rank}\n"
                    f"{score:.3f} | {status_text}"
                )
            else:
                title = (
                    f"Top-{rank}\n"
                    f"{score:.3f}"
                )

        else:

            if status_text:
                title = (
                    f"Top-{rank}\n"
                    f"{status_text}"
                )
            else:
                title = f"Top-{rank}"

        ax.set_title(
            title,
            fontsize=8,
        )

    # ==================================================
    # 7. Legend
    # ==================================================
    colors = [
        [1.00, 0.10, 0.10],
        [0.10, 0.70, 1.00],
        [0.10, 0.90, 0.25],
        [1.00, 0.80, 0.10],
        [0.90, 0.20, 0.80],
        [0.20, 0.90, 0.90],
    ]

    legend_handles = []

    # Retrieval 결과에서 실제로 표시되는 organ 기준
    for idx, organ_id in enumerate(
        overlay_organs
    ):

        legend_handles.append(
            Patch(
                facecolor=colors[
                    idx % len(colors)
                ],
                edgecolor="none",
                label=f"Organ {int(organ_id)}",
            )
        )

    # bbox legend
    if vis_bbox is not None:

        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                fill=False,
                edgecolor="yellow",
                linewidth=2,
                label="Query ROI",
            )
        )

    if len(legend_handles) > 0:

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(
                len(legend_handles),
                5,
            ),
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    # ==================================================
    # 8. Figure title
    # ==================================================
    organ_tag = "_".join(
        str(int(o))
        for o in selected_organs
    )

    if query_bbox is not None:

        fig.suptitle(
            f"Query ROI | Regions: {organ_text}",
            fontsize=11,
            y=0.94,
        )

    else:

        fig.suptitle(
            f"Query region(s): {organ_text}",
            fontsize=11,
            y=0.94,
        )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.78,
        bottom=0.02,
        wspace=0.02,
    )

    # ==================================================
    # 9. Save
    # ==================================================
    if save_name is None:

        if query_bbox is not None:

            x1, y1, x2, y2 = query_bbox

            save_name = (
                f"{query_slide_name}"
                f"_bbox_{x1}_{y1}_{x2}_{y2}"
                f"_organs_{organ_tag}"
                f"_top{len(topk_filenames)}"
            )

        else:

            save_name = (
                f"{query_slide_name}"
                f"_organs_{organ_tag}"
                f"_top{len(topk_filenames)}"
            )

        if highlight_intr_organs:
            save_name += "_intr"

    save_path = os.path.join(
        save_dir,
        save_name + ".png",
    )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    if vis:
        plt.show()

    plt.close(fig)

    print(f"Saved: {save_path}")

    return save_path

def make_rect_mask_from_organs(
    orig_seg,
    selected_organs,
    box_size,
    center_offset=(0, 0),
    patch_grid_size=37,
):
    """
    관심 장기의 centroid를 기준으로 rectangular mask 생성.

    Parameters
    ----------
    orig_seg : torch.Tensor
        [H, W] 또는 [1, H, W]

    selected_organs : list
        bbox 중심을 정의할 anatomical region IDs.

    box_size : int or tuple
        bbox 크기. patch 단위.

        int:
            3 -> 3 x 3 patches
            12 -> 12 x 12 patches

        tuple:
            (width, height)
            (12, 6) -> width 12 patches, height 6 patches
            (6, 12) -> width 6 patches, height 12 patches

    center_offset : tuple
        (dx, dy), patch 단위.

        dx > 0 : right
        dx < 0 : left
        dy > 0 : down
        dy < 0 : up

        이동은 항상 원래 organ centroid를 기준으로 함.

    patch_grid_size : int
        ViT patch grid 크기.
        예: 518 / 37 = 14 pixels per patch.

    Returns
    -------
    rect_mask
        orig_seg와 동일 spatial resolution의 rectangular mask.

    bbox
        (x1, y1, x2, y2)

    center
        실제 이동된 bbox center (cx, cy), pixel coordinate.
    """

    # --------------------------------------------------
    # 1. Segmentation -> 2D
    # --------------------------------------------------
    if orig_seg.ndim == 3:
        seg_2d = orig_seg[0]
    else:
        seg_2d = orig_seg

    H, W = seg_2d.shape

    # --------------------------------------------------
    # 2. Anchor organ mask
    # --------------------------------------------------
    selected_organs = torch.as_tensor(
        selected_organs,
        device=seg_2d.device,
        dtype=seg_2d.dtype,
    )

    organ_mask = torch.isin(
        seg_2d,
        selected_organs,
    )

    ys, xs = torch.where(organ_mask)

    if len(xs) == 0:
        raise ValueError(
            f"Selected organs {selected_organs.tolist()} "
            "are not present in the query slice."
        )

    # --------------------------------------------------
    # 3. Original organ centroid
    # --------------------------------------------------
    cx_anchor = xs.float().mean().item()
    cy_anchor = ys.float().mean().item()

    # --------------------------------------------------
    # 4. Patch size
    # --------------------------------------------------
    patch_w = W / float(patch_grid_size)
    patch_h = H / float(patch_grid_size)

    # --------------------------------------------------
    # 5. Center shift
    #
    # center_offset = (dx, dy), patch units
    # --------------------------------------------------
    dx, dy = center_offset

    cx = cx_anchor + dx * patch_w
    cy = cy_anchor + dy * patch_h

    # --------------------------------------------------
    # 6. bbox size
    #
    # int:
    #   12 -> 12 x 12
    #
    # tuple:
    #   (12, 6) -> width 12, height 6
    # --------------------------------------------------
    if isinstance(box_size, (tuple, list)):

        if len(box_size) != 2:
            raise ValueError(
                "box_size tuple/list must be "
                "(width_in_patches, height_in_patches)."
            )

        box_w_patches = float(box_size[0])
        box_h_patches = float(box_size[1])

    else:

        box_w_patches = float(box_size)
        box_h_patches = float(box_size)

    if box_w_patches <= 0 or box_h_patches <= 0:
        raise ValueError(
            f"box_size must be positive, got {box_size}"
        )

    # patch -> pixel
    bbox_w = int(
        round(box_w_patches * patch_w)
    )

    bbox_h = int(
        round(box_h_patches * patch_h)
    )

    # 최소 1 pixel
    bbox_w = max(1, bbox_w)
    bbox_h = max(1, bbox_h)

    # bbox가 image보다 클 경우 방지
    bbox_w = min(bbox_w, W)
    bbox_h = min(bbox_h, H)

    # --------------------------------------------------
    # 7. bbox coordinates
    # --------------------------------------------------
    x1 = int(
        round(cx - bbox_w / 2)
    )

    y1 = int(
        round(cy - bbox_h / 2)
    )

    # --------------------------------------------------
    # image boundary에서도 bbox 크기를 유지하도록
    # bbox 전체를 안쪽으로 이동
    # --------------------------------------------------
    x1 = max(
        0,
        min(x1, W - bbox_w),
    )

    y1 = max(
        0,
        min(y1, H - bbox_h),
    )

    x2 = x1 + bbox_w
    y2 = y1 + bbox_h

    # --------------------------------------------------
    # 실제 bbox center
    # --------------------------------------------------
    actual_cx = (
        x1 + x2
    ) / 2.0

    actual_cy = (
        y1 + y2
    ) / 2.0

    # --------------------------------------------------
    # 8. Rectangular mask
    # --------------------------------------------------
    mask_2d = torch.zeros(
        (H, W),
        dtype=torch.float32,
        device=seg_2d.device,
    )

    mask_2d[
        y1:y2,
        x1:x2
    ] = 1.0

    # --------------------------------------------------
    # 원래 input dimension 유지
    # --------------------------------------------------
    if orig_seg.ndim == 3:
        rect_mask = mask_2d.unsqueeze(0)
    else:
        rect_mask = mask_2d

    return (
        rect_mask,
        (x1, y1, x2, y2),
        (actual_cx, actual_cy),
    )


def get_organs_in_bbox(
    seg,
    bbox,
    exclude_background=True,
    min_pixels=20,
):
    """
    bbox 내부에 포함된 organ id 목록 반환.

    Parameters
    ----------
    seg : np.ndarray [H, W]
    bbox : tuple (x1, y1, x2, y2)
    min_pixels : int
        bbox 내부에서 너무 작은 fragment는 제거

    Returns
    -------
    organ_ids : np.ndarray
    """
    x1, y1, x2, y2 = bbox

    H, W = seg.shape
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))

    crop = seg[y1:y2, x1:x2]

    organ_ids, counts = np.unique(crop, return_counts=True)

    kept = []
    for organ_id, count in zip(organ_ids, counts):
        if exclude_background and organ_id == 0:
            continue
        if count >= min_pixels:
            kept.append(int(organ_id))

    return np.asarray(kept, dtype=np.int64)


def draw_bbox_on_axis(ax, bbox, color="yellow", linewidth=2):
    """
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)

def visualize_multi_query_topk2(
    query_slide_name,
    topk_filenames,
    topk_sim,
    root_dir,
    selected_organs,
    query_bbox=None,
    bbox_source_shape=None,
    intr_organs=None,
    topk=10,
    save_dir="./results/retrieval_visualization",
    save_name=None,
    alpha=0.45,
    bbox_alpha=0.25,
    dpi=300,
    show_score=True,
    show_missing=True,
    vis=False,
):
    """
    Query bbox 기반 retrieval visualization.

    Query:
        실제 query selection인 bbox만 표시.

    Retrieval results:
        intr_organs만 segmentation 기반으로 overlay.

    Parameters
    ----------
    query_slide_name : str
        Query slice 이름.

    topk_filenames : list or np.ndarray
        Retrieval 결과 slice key.

    topk_sim : torch.Tensor or np.ndarray
        Retrieval similarity score.

    root_dir : str
        Dataset root directory.

    selected_organs : torch.Tensor, np.ndarray, or list
        Query bbox 내부에 포함된 anatomical region IDs.
        Retrieval 설정 정보로만 사용하며 overlay에는 사용하지 않음.

    query_bbox : tuple, optional
        (x1, y1, x2, y2)
        bbox_source_shape 좌표계에서 정의된 query ROI.

    bbox_source_shape : tuple, optional
        query_bbox가 정의된 spatial shape.
        예: orig_seg.shape[-2:] == (518, 518)

    intr_organs : optional
        Retrieval 결과에서 관심 있게 확인할 anatomical regions.
        Top-k 결과에서 해당 organ들만 색칠하고 presence를 표시.
        None이면 selected_organs 사용.

    topk : int
        표시할 retrieval 결과 수.

    alpha : float
        Retrieval 결과의 intr_organs overlay alpha.

    bbox_alpha : float
        Query bbox overlay alpha.

    dpi : int
        Figure resolution.

    show_score : bool
        Similarity score 표시 여부.

    show_missing : bool
        intr_organs의 presence/missing 여부 표시.

    vis : bool
        True이면 plt.show().
    """

    from matplotlib.patches import Patch, Rectangle

    os.makedirs(save_dir, exist_ok=True)

    # ==================================================
    # 0. Input 정리
    # ==================================================
    selected_organs_vis = np.asarray(
        to_numpy_organs(selected_organs)
    ).reshape(-1)

    if intr_organs is None:
        intr_organs = selected_organs_vis.copy()
    else:
        intr_organs = np.asarray(
            to_numpy_organs(intr_organs)
        ).reshape(-1)

    topk_filenames = to_numpy_filenames(topk_filenames)
    topk_scores = to_numpy_scores(topk_sim)

    topk_filenames = topk_filenames[:topk]
    topk_scores = topk_scores[:topk]

    # ==================================================
    # 1. Query load
    # ==================================================
    q_img_path = make_image_path(
        root_dir,
        query_slide_name,
    )

    q_img = load_ct_image_for_vis(
        q_img_path
    )

    # ==================================================
    # 2. bbox coordinate conversion
    #
    # query_bbox:
    #     retrieval / orig_seg 좌표계
    #
    # vis_bbox:
    #     visualization image 좌표계
    # ==================================================
    vis_bbox = None

    if query_bbox is not None:

        x1, y1, x2, y2 = query_bbox

        if bbox_source_shape is not None:

            src_h, src_w = bbox_source_shape
            dst_h, dst_w = q_img.shape[:2]

            scale_x = dst_w / float(src_w)
            scale_y = dst_h / float(src_h)

            vis_bbox = (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            )

        else:

            vis_bbox = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )

    # ==================================================
    # 3. Figure
    # ==================================================
    n_cols = 1 + len(topk_filenames)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(2.8 * n_cols, 3.4),
        dpi=dpi,
    )

    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # ==================================================
    # 4. Query panel
    # ==================================================
    if q_img.ndim == 2:
        q_img_vis = np.stack(
            [q_img, q_img, q_img],
            axis=-1,
        )
    else:
        q_img_vis = q_img.copy()

    axes[0].imshow(
        q_img_vis,
        interpolation="nearest",
    )

    # --------------------------------------------------
    # Query bbox overlay
    # --------------------------------------------------
    if vis_bbox is not None:

        x1, y1, x2, y2 = vis_bbox

        # bbox 내부
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor="yellow",
            edgecolor="yellow",
            linewidth=2.5,
            alpha=bbox_alpha,
            zorder=20,
        )

        axes[0].add_patch(rect)

        # bbox border
        rect_border = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="yellow",
            linewidth=2.0,
            zorder=21,
        )

        axes[0].add_patch(rect_border)

    organs_text = ", ".join(
        str(int(o))
        for o in selected_organs_vis
    )

    axes[0].set_title(
        f"Query\n{query_slide_name}",
        fontsize=8,
    )

    # ==================================================
    # 5. Top-k panels
    # ==================================================
    for rank, (db_key, score) in enumerate(
        zip(topk_filenames, topk_scores),
        start=1,
    ):

        ax = axes[rank]

        db_img_path = make_image_path(
            root_dir,
            db_key,
        )

        db_seg_path = make_seg_path(
            root_dir,
            db_key,
        )

        db_img = load_ct_image_for_vis(
            db_img_path
        )

        db_seg = load_seg_for_vis(
            db_seg_path
        )

        # --------------------------------------------------
        # Retrieval 결과에서는 intr_organs만 색칠
        # selected_organs는 색칠하지 않음
        # --------------------------------------------------
        db_overlay = overlay_selected_organs(
            image=db_img,
            seg=db_seg,
            selected_organs=intr_organs,
            alpha=alpha,
        )

        # --------------------------------------------------
        # Presence 역시 intr_organs 기준
        # --------------------------------------------------
        organ_presence = get_selected_organ_presence(
            db_seg,
            intr_organs,
        )

        if show_missing:

            status_text = " ".join(
                f"{int(organ_id)}"
                f"{'✓' if present else '✗'}"
                for organ_id, present
                in organ_presence.items()
            )

        else:

            status_text = " ".join(
                str(int(organ_id))
                for organ_id, present
                in organ_presence.items()
                if present
            )

        ax.imshow(
            db_overlay,
            interpolation="nearest",
        )

        if show_score:

            if len(status_text) > 0:

                title = (
                    f"Top-{rank}\n"
                    f"{score:.3f} | {status_text}"
                )

            else:

                title = (
                    f"Top-{rank}\n"
                    f"{score:.3f}"
                )

        else:

            if len(status_text) > 0:

                title = (
                    f"Top-{rank}\n"
                    f"{status_text}"
                )

            else:

                title = f"Top-{rank}"

        ax.set_title(
            title,
            fontsize=8,
        )

    # ==================================================
    # 6. Legend
    # ==================================================
    colors = [
        [1.00, 0.10, 0.10],
        [0.10, 0.70, 1.00],
        [0.10, 0.90, 0.25],
        [1.00, 0.80, 0.10],
        [0.90, 0.20, 0.80],
        [0.20, 0.90, 0.90],
    ]

    legend_handles = []

    # intr_organs만 legend에 표시
    for idx, organ_id in enumerate(
        intr_organs
    ):

        legend_handles.append(
            Patch(
                facecolor=colors[
                    idx % len(colors)
                ],
                edgecolor="none",
                label=f"Organ {int(organ_id)}",
            )
        )

    # Query bbox legend
    if vis_bbox is not None:

        legend_handles.append(
            Patch(
                facecolor="yellow",
                edgecolor="yellow",
                alpha=bbox_alpha,
                label="Query ROI",
            )
        )

    if len(legend_handles) > 0:

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(
                len(legend_handles),
                5,
            ),
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 1.03),
        )

    # ==================================================
    # 7. Figure title
    # ==================================================
    organ_tag = "_".join(
        str(int(o))
        for o in selected_organs_vis
    )

    if query_bbox is not None:

        fig.suptitle(
            f"Query ROI | Included organs: {organs_text}",
            fontsize=11,
            y=0.95,
        )

    else:

        fig.suptitle(
            f"Query organ(s): {organs_text}",
            fontsize=11,
            y=0.95,
        )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.77,
        bottom=0.02,
        wspace=0.02,
    )

    # ==================================================
    # 8. Save
    # ==================================================
    if save_name is None:

        if query_bbox is not None:

            x1, y1, x2, y2 = query_bbox

            save_name = (
                f"{query_slide_name}"
                f"_bbox_{x1}_{y1}_{x2}_{y2}"
                f"_organs_{organ_tag}"
                f"_top{len(topk_filenames)}"
            )

        else:

            save_name = (
                f"{query_slide_name}"
                f"_organs_{organ_tag}"
                f"_top{len(topk_filenames)}"
            )

    save_path = os.path.join(
        save_dir,
        save_name + ".png",
    )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    if vis:
        plt.show()

    plt.close(fig)

    print(f"Saved: {save_path}")

    return save_path
#
#
# def visualize_multi_query_topk2(
#     query_slide_name,
#     topk_filenames,
#     topk_sim,
#     root_dir,
#     selected_organs,
#     query_bbox=None,
#     bbox_source_shape=None,
#     intr_organs=None,
#     topk=10,
#     save_dir="./results/retrieval_visualization",
#     save_name=None,
#     alpha=0.45,
#     bbox_alpha=0.25,
#     dpi=300,
#     show_score=True,
#     show_missing=True,
#     vis=False,
# ):
#     """
#     Query bbox 기반 retrieval visualization.
#
#     Query:
#         실제 선택 영역인 bbox 전체를 red overlay로 표시.
#
#     Retrieval results:
#         bbox 내부에 포함된 anatomical regions인 selected_organs를
#         segmentation 기반으로 overlay.
#
#     Parameters
#     ----------
#     query_slide_name : str
#         Query slice 이름.
#
#     topk_filenames : list or np.ndarray
#         Retrieval 결과 slice key.
#
#     topk_sim : torch.Tensor or np.ndarray
#         Retrieval similarity score.
#
#     root_dir : str
#         Dataset root directory.
#
#     selected_organs : torch.Tensor, np.ndarray, or list
#         Query bbox 내부에 포함된 anatomical region IDs.
#         Retrieval 결과에서 해당 region들을 색칠하는 데 사용.
#
#     query_bbox : tuple, optional
#         (x1, y1, x2, y2)
#         bbox_source_shape 좌표계에서 정의된 query ROI.
#
#     bbox_source_shape : tuple, optional
#         query_bbox가 정의된 spatial shape.
#         예: orig_seg.shape[-2:] == (518, 518)
#
#     intr_organs : optional
#         Retrieval 결과에서 presence 여부를 확인할 region.
#         None이면 selected_organs 사용.
#
#     topk : int
#         표시할 retrieval 결과 수.
#
#     alpha : float
#         Retrieval 결과의 organ segmentation overlay alpha.
#
#     bbox_alpha : float
#         Query bbox 내부 red overlay의 alpha.
#
#     dpi : int
#         Figure resolution.
#
#     show_score : bool
#         Similarity score 표시 여부.
#
#     show_missing : bool
#         관심 organ의 presence/missing 여부 표시.
#
#     vis : bool
#         True이면 plt.show().
#     """
#
#     from matplotlib.patches import Patch, Rectangle
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     # ==================================================
#     # 0. Input 정리
#     # ==================================================
#     selected_organs_vis = np.asarray(
#         to_numpy_organs(selected_organs)
#     ).reshape(-1)
#
#     if intr_organs is None:
#         intr_organs = selected_organs_vis.copy()
#     else:
#         intr_organs = np.asarray(
#             to_numpy_organs(intr_organs)
#         ).reshape(-1)
#
#     topk_filenames = to_numpy_filenames(topk_filenames)
#     topk_scores = to_numpy_scores(topk_sim)
#
#     topk_filenames = topk_filenames[:topk]
#     topk_scores = topk_scores[:topk]
#
#     # ==================================================
#     # 1. Query load
#     # ==================================================
#     q_img_path = make_image_path(
#         root_dir,
#         query_slide_name,
#     )
#
#     q_img = load_ct_image_for_vis(
#         q_img_path
#     )
#
#     # ==================================================
#     # 2. bbox coordinate conversion
#     #
#     # query_bbox:
#     #     retrieval / orig_seg 좌표계
#     #
#     # vis_bbox:
#     #     visualization image 좌표계
#     # ==================================================
#     vis_bbox = None
#
#     if query_bbox is not None:
#
#         x1, y1, x2, y2 = query_bbox
#
#         if bbox_source_shape is not None:
#
#             src_h, src_w = bbox_source_shape
#
#             dst_h, dst_w = q_img.shape[:2]
#
#             scale_x = dst_w / float(src_w)
#             scale_y = dst_h / float(src_h)
#
#             vis_bbox = (
#                 int(round(x1 * scale_x)),
#                 int(round(y1 * scale_y)),
#                 int(round(x2 * scale_x)),
#                 int(round(y2 * scale_y)),
#             )
#
#         else:
#
#             vis_bbox = (
#                 int(x1),
#                 int(y1),
#                 int(x2),
#                 int(y2),
#             )
#
#     # ==================================================
#     # 3. Figure
#     # ==================================================
#     n_cols = 1 + len(topk_filenames)
#
#     fig, axes = plt.subplots(
#         1,
#         n_cols,
#         figsize=(2.8 * n_cols, 3.4),
#         dpi=dpi,
#     )
#
#     axes = np.asarray(axes).reshape(-1)
#
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.axis("off")
#
#     # ==================================================
#     # 4. Query panel
#     # ==================================================
#     #
#     # Query에서는 anatomical segmentation을 표시하지 않음.
#     # 실제 retrieval selection인 bbox 자체만 표시.
#     # ==================================================
#     q_img = np.stack([q_img, q_img, q_img], axis=-1)
#     axes[0].imshow(
#         q_img,
#         interpolation="nearest",
#     )
#
#     # --------------------------------------------------
#     # Query bbox
#     #
#     # bbox 내부 전체를 red + alpha로 overlay
#     # --------------------------------------------------
#     if vis_bbox is not None:
#
#         x1, y1, x2, y2 = vis_bbox
#
#         rect = Rectangle(
#             (x1, y1),
#             x2 - x1,
#             y2 - y1,
#             facecolor="yellow",
#             edgecolor="yellow",
#             linewidth=2.5,
#             alpha=bbox_alpha,
#             zorder=20,
#         )
#
#         axes[0].add_patch(rect)
#
#         # alpha 때문에 border까지 흐려지는 것을 방지하기 위해
#         # bbox outline을 한 번 더 그림
#         rect_border = Rectangle(
#             (x1, y1),
#             x2 - x1,
#             y2 - y1,
#             fill=False,
#             edgecolor="yellow",
#             linewidth=2.0,
#             zorder=21,
#         )
#
#         axes[0].add_patch(rect_border)
#
#     organs_text = ", ".join(
#         str(int(o))
#         for o in selected_organs_vis
#     )
#
#     axes[0].set_title(
#         f"Query\n{query_slide_name}",
#         fontsize=8,
#     )
#
#     # ==================================================
#     # 5. Top-k panels
#     # ==================================================
#     for rank, (db_key, score) in enumerate(
#         zip(topk_filenames, topk_scores),
#         start=1,
#     ):
#
#         ax = axes[rank]
#
#         db_img_path = make_image_path(
#             root_dir,
#             db_key,
#         )
#
#         db_seg_path = make_seg_path(
#             root_dir,
#             db_key,
#         )
#
#         db_img = load_ct_image_for_vis(
#             db_img_path
#         )
#
#         db_seg = load_seg_for_vis(
#             db_seg_path
#         )
#
#         # --------------------------------------------------
#         # Query bbox 내부에 포함된 anatomical regions를
#         # retrieval 결과에서 segmentation으로 표시
#         # --------------------------------------------------
#         db_overlay = overlay_selected_organs(
#             image=db_img,
#             seg=db_seg,
#             selected_organs=selected_organs_vis,
#             alpha=alpha,
#         )
#
#         # --------------------------------------------------
#         # Presence
#         # --------------------------------------------------
#         organ_presence = get_selected_organ_presence(
#             db_seg,
#             intr_organs,
#         )
#
#         if show_missing:
#
#             status_text = " ".join(
#                 f"{int(organ_id)}"
#                 f"{'✓' if present else '✗'}"
#                 for organ_id, present
#                 in organ_presence.items()
#             )
#
#         else:
#
#             status_text = " ".join(
#                 str(int(organ_id))
#                 for organ_id, present
#                 in organ_presence.items()
#                 if present
#             )
#
#         ax.imshow(
#             db_overlay,
#             interpolation="nearest",
#         )
#
#         if show_score:
#
#             if len(status_text) > 0:
#
#                 title = (
#                     f"Top-{rank}\n"
#                     f"{score:.3f} | {status_text}"
#                 )
#
#             else:
#
#                 title = (
#                     f"Top-{rank}\n"
#                     f"{score:.3f}"
#                 )
#
#         else:
#
#             if len(status_text) > 0:
#
#                 title = (
#                     f"Top-{rank}\n"
#                     f"{status_text}"
#                 )
#
#             else:
#
#                 title = f"Top-{rank}"
#
#         ax.set_title(
#             title,
#             fontsize=8,
#         )
#
#     # ==================================================
#     # 6. Legend
#     # ==================================================
#     colors = [
#         [1.00, 0.10, 0.10],
#         [0.10, 0.70, 1.00],
#         [0.10, 0.90, 0.25],
#         [1.00, 0.80, 0.10],
#         [0.90, 0.20, 0.80],
#         [0.20, 0.90, 0.90],
#     ]
#
#     legend_handles = []
#
#     # Retrieval result의 organ overlay legend
#     for idx, organ_id in enumerate(
#         selected_organs_vis
#     ):
#
#         legend_handles.append(
#             Patch(
#                 facecolor=colors[
#                     idx % len(colors)
#                 ],
#                 edgecolor="none",
#                 label=f"Organ {int(organ_id)}",
#             )
#         )
#
#     # Query bbox legend
#     if vis_bbox is not None:
#
#         legend_handles.append(
#             Patch(
#                 facecolor="red",
#                 edgecolor="red",
#                 alpha=bbox_alpha,
#                 label="Query ROI",
#             )
#         )
#
#     if len(legend_handles) > 0:
#
#         fig.legend(
#             handles=legend_handles,
#             loc="upper center",
#             ncol=min(
#                 len(legend_handles),
#                 5,
#             ),
#             fontsize=9,
#             frameon=False,
#             bbox_to_anchor=(0.5, 1.03),
#         )
#
#     # ==================================================
#     # 7. Figure title
#     # ==================================================
#     organ_tag = "_".join(
#         str(int(o))
#         for o in selected_organs_vis
#     )
#
#     if query_bbox is not None:
#
#         fig.suptitle(
#             f"Query ROI | Included organs: {organs_text}",
#             fontsize=11,
#             y=0.95,
#         )
#
#     else:
#
#         fig.suptitle(
#             f"Query organ(s): {organs_text}",
#             fontsize=11,
#             y=0.95,
#         )
#
#     fig.subplots_adjust(
#         left=0.01,
#         right=0.99,
#         top=0.77,
#         bottom=0.02,
#         wspace=0.02,
#     )
#
#     # ==================================================
#     # 8. Save
#     # ==================================================
#     if save_name is None:
#
#         if query_bbox is not None:
#
#             x1, y1, x2, y2 = query_bbox
#
#             save_name = (
#                 f"{query_slide_name}"
#                 f"_bbox_{x1}_{y1}_{x2}_{y2}"
#                 f"_organs_{organ_tag}"
#                 f"_top{len(topk_filenames)}"
#             )
#
#         else:
#
#             save_name = (
#                 f"{query_slide_name}"
#                 f"_organs_{organ_tag}"
#                 f"_top{len(topk_filenames)}"
#             )
#
#     save_path = os.path.join(
#         save_dir,
#         save_name + ".png",
#     )
#
#     fig.savefig(
#         save_path,
#         dpi=dpi,
#         bbox_inches="tight",
#         pad_inches=0.02,
#     )
#
#     if vis:
#         plt.show()
#
#     plt.close(fig)
#
#     print(f"Saved: {save_path}")
#
#     return save_path