import json
import numpy as np
import os
import nibabel as nib
import glob
import re
import matplotlib.pyplot as plt
from PIL import Image
import tqdm

def normalize_ct(slice_img, hu_min=-963, hu_max=1053):
# def normalize_ct(slice_img, hu_min=-1000, hu_max=400):
    slice_img = np.clip(slice_img, hu_min, hu_max)
    slice_img = (slice_img - hu_min) / (hu_max - hu_min)
    slice_img = (slice_img * 255).astype(np.uint8)
    return slice_img

def pad_to_square_np(img, pad_value=0):
    """
    2D image를 정사각형으로 padding.
    중심 정렬 방식.
    """
    h, w = img.shape

    if h == w:
        return img

    size = max(h, w)

    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    padded = np.pad(
        img,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded


def save_organ_crop(
    image_np,
    seg_array,
    organ_id,
    bbox,
    save_dir,
    filename,
    margin_ratio=0.1,
    apply_mask=True,
    pad_to_square=True,
    pad_value=0,
):
    """
    image_np: 2D slice image array
    seg_array: 2D segmentation map
    organ_id: target organ id
    bbox: [x1, y1, x2, y2]
    save_dir: crop 저장 폴더
    filename: slice filename
    margin_ratio: bbox 주변 margin 비율
                  -1이면 전체 이미지 사용
    apply_mask: True이면 organ 외부 영역을 0으로 처리
    pad_to_square: True이면 crop 후 정사각형 padding
    pad_value: padding 값
    """

    os.makedirs(save_dir, exist_ok=True)

    H, W = image_np.shape
    x1, y1, x2, y2 = map(int, bbox)

    bw = x2 - x1
    bh = y2 - y1

    if margin_ratio == -1:
        x1m, y1m = 0, 0
        x2m, y2m = W, H
    else:
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)

        x1m = max(0, x1 - mx)
        y1m = max(0, y1 - my)
        x2m = min(W, x2 + mx)
        y2m = min(H, y2 + my)

    crop = image_np[y1m:y2m, x1m:x2m]

    if apply_mask:
        mask = (seg_array == organ_id).astype(np.uint8)
        mask_crop = mask[y1m:y2m, x1m:x2m]
        crop = crop * mask_crop

    if pad_to_square:
        crop = pad_to_square_np(crop, pad_value=pad_value)

    crop_name = f"{filename}__organ_{int(organ_id)}.png"
    crop_path = os.path.join(save_dir, crop_name)

    Image.fromarray(crop).save(crop_path)

    return crop_path

def compute_organ_geometry(seg_array, organ_id):
    """
    seg_array: 2D segmentation map, shape [H, W]
    organ_id: target organ id

    Computes organ geometry using both:
    1) whole-slice normalization
    2) foreground-organ-region normalization
    """

    H, W = seg_array.shape

    organ_mask = seg_array == organ_id
    foreground_mask = seg_array > 0

    if organ_mask.sum() == 0:
        return None

    if foreground_mask.sum() == 0:
        return None

    # -------------------------
    # Organ mask geometry
    # -------------------------
    ys, xs = np.where(organ_mask)

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    area = int(organ_mask.sum())

    cx = float(xs.mean())
    cy = float(ys.mean())

    bbox = [int(x1), int(y1), int(x2), int(y2)]
    centroid = [cx, cy]

    bbox_area = int((x2 - x1) * (y2 - y1))

    # -------------------------
    # Whole-slice relative geometry
    # -------------------------
    slice_area = int(H * W)

    slice_relative_centroid = [
        float(cx / W),
        float(cy / H),
    ]

    slice_relative_area = float(area / slice_area)
    slice_relative_bbox_area = float(bbox_area / slice_area)

    # -------------------------
    # Foreground-organ-region geometry
    # -------------------------
    fg_ys, fg_xs = np.where(foreground_mask)

    fg_x1, fg_x2 = fg_xs.min(), fg_xs.max() + 1
    fg_y1, fg_y2 = fg_ys.min(), fg_ys.max() + 1

    fg_w = max(fg_x2 - fg_x1, 1)
    fg_h = max(fg_y2 - fg_y1, 1)

    foreground_area = int(foreground_mask.sum())
    foreground_bbox = [int(fg_x1), int(fg_y1), int(fg_x2), int(fg_y2)]
    foreground_bbox_area = int(fg_w * fg_h)

    foreground_relative_centroid = [
        float((cx - fg_x1) / fg_w),
        float((cy - fg_y1) / fg_h),
    ]

    foreground_relative_area = float(area / max(foreground_area, 1))
    foreground_relative_bbox_area = float(bbox_area / max(foreground_bbox_area, 1))

    return {
        "bbox": bbox,
        "centroid": centroid,
        "area": area,
        "bbox_area": bbox_area,

        # relative to whole slice
        "slice_relative_centroid": slice_relative_centroid,
        "slice_relative_area": slice_relative_area,
        "slice_relative_bbox_area": slice_relative_bbox_area,

        # relative to all segmented organ regions
        "foreground_bbox": foreground_bbox,
        "foreground_area": foreground_area,
        "foreground_bbox_area": foreground_bbox_area,
        "foreground_relative_centroid": foreground_relative_centroid,
        "foreground_relative_area": foreground_relative_area,
        "foreground_relative_bbox_area": foreground_relative_bbox_area,
    }

def normalize_name(name: str):
    original = name.strip()

    # vertebrae 형태는 대소문자 포함 그대로 유지하기 위해 lower() 안 씀
    low = original.lower()

    # 1️⃣ vertebrae 중복 숫자 제거 (예: vertebrae l5 5 → vertebrae L5)
    m_dup = re.match(r'vertebrae\s+([a-z])(\d+)\s+\2$', low)
    if m_dup:
        letter, number = m_dup.groups()
        return f"vertebrae_{letter.upper()}{number}"

    # 2️⃣ vertebrae 정상 형태 그대로 유지
    if re.match(r'vertebrae\s+[A-Z]\d+$', original):
        original = original.replace(' ', '_')
        return original # 그대로 반환
    if re.match(r'vertebrae\s+[a-z]\d+$', original):
        # 소문자면 대문자로 바꾸기만
        letter_num = re.findall(r'[a-z]\d+', original)[0]
        return f"vertebrae_{letter_num.upper()}"

    # 3️⃣ 일반 기관명 처리
    name = low
    has_left = 'left' in name
    has_right = 'right' in name
    num_match = re.search(r'\d+', name)

    parts = [p for p in name.split() if p not in ['left', 'right']]

    if num_match:
        num = num_match.group()
        parts = [p for p in parts if p != num]
        if has_left:
            normalized = f"{' '.join(parts)} left {num}"
        elif has_right:
            normalized = f"{' '.join(parts)} right {num}"
        else:
            normalized = f"{' '.join(parts)} {num}"
    else:
        if has_left:
            normalized = f"{' '.join(parts)} left"
        elif has_right:
            normalized = f"{' '.join(parts)} right"
        else:
            normalized = ' '.join(parts)
    if normalized == 'bladder':
        normalized = 'urinary bladder'

    # if re.match(r'vertebrae', normalized):
    normalized = normalized.replace(' ', '_')
    return normalized.strip()

##########################################################
### TotalSegmentator_label_index_mapping load
##########################################################



seg_type = 'totalseg'
datasetname = 'totalsegmentator_retrieval'


with open(os.path.join(f'./datasets/{datasetname}/TotalSegmentator_label_index_mapping.json'), 'r') as fp:
    label = json.load(fp)

label["idx2label"] = {int(k): v for k, v in label["idx2label"].items()}
label["label2idx"] = {k: int(v) for k, v in label["label2idx"].items()}
totalseg_label2idx = label['label2idx']
totalseg_idx2label = label['idx2label']
totalseg = np.array(list(totalseg_idx2label.values()))

with open(f'./datasets/{datasetname}/label_vista3d_dict.json', 'rb') as f:
    label_dict = json.load(f)

idx2label = {}
for k, v in label_dict.items():
    idx2label[v] = k
vista3dseg = np.array(list(idx2label.values()))
normalized_dict = {n : normalize_name(n) for n in vista3dseg}
name_to_vista3d_id = {}
for k, v in label_dict.items():
    name_to_vista3d_id[normalized_dict[k]] = v

vista3d_id_to_name = {}
for k, v in label_dict.items():
    vista3d_id_to_name[v] = normalized_dict[k]

totalseg2vista_mapper = {}
for k, v in totalseg_label2idx.items():
    totalseg2vista_mapper[v] = name_to_vista3d_id[k]

vista2totalseg_mapper = {}
for k, v in totalseg_label2idx.items():
    vista2totalseg_mapper[name_to_vista3d_id[k]] = v

database_index = {}

# -------------------------
# 1. Meta information
# -------------------------
database_index["meta"] = {}

database_index["meta"]["dataset"] = "TotalSegmentator_v2"
database_index["meta"]["index_version"] = "v1"
database_index["meta"]["primary_label_space"] = "totalseg"
database_index["meta"]["seg_sources"] = seg_type

# image / slice setting
database_index["meta"]["image_size"] = None  # 예: [512, 512]
database_index["meta"]["spacing"] = None     # 필요 시 예: [sx, sy]

# -------------------------
# 2. ID mapping dictionaries
# -------------------------
database_index["meta"]["id_maps"] = {}
database_index["meta"]["id_maps"]["totalseg_id_to_name"] = totalseg_idx2label
database_index["meta"]["id_maps"]["name_to_totalseg_id"] = totalseg_label2idx

# -------------------------
# 3. Slice-level index
# -------------------------
database_index["slices"] = {}
db_imgs = glob.glob(f'./datasets/{datasetname}/gallery/image/*')
np.random.shuffle(db_imgs)
for i in tqdm.tqdm(range(len(db_imgs))):
    db_img_path = db_imgs[i]
    gt_seg_query_path = db_img_path.replace('/image/', '/seg_gt/').replace('.png', '.npy')
    vista_seg_query_path = db_img_path.replace('/image/', '/seg_vista3d/').replace('.png', '.npy')
    filename = os.path.splitext(os.path.basename(db_img_path))[0]

    if seg_type == 'totalseg':
        seg_query_path = gt_seg_query_path
    elif seg_type == 'vista3d':
        seg_query_path = vista_seg_query_path
    else:
        assert False

    # image = Image.open(db_img_path).convert("L")
    image_np = np.load(db_img_path)
    image_np = normalize_ct(image_np)

    _seg = np.load(seg_query_path)
    unique_organ = np.unique(_seg)
    gt_organ_idx = unique_organ[unique_organ!=0]
    gt_organ_names = [totalseg_idx2label[int(i)] for i in gt_organ_idx]


    database_index["slices"][filename] = {
        "image_path": db_img_path,
        "seg_type" : seg_type,
        "seg_paths": seg_query_path,
        "organ_list": gt_organ_names,
        "organ_idx": gt_organ_idx.tolist(),
        "organ_info": {}
    }


    for organ_id in gt_organ_idx:
        organ_id = int(organ_id)
        organ_name = totalseg_idx2label[organ_id]
        geometry = compute_organ_geometry(_seg, organ_id)

        if geometry is None:
            continue
        crop_path = save_organ_crop(
            image_np=image_np,
            seg_array=_seg,
            organ_id=organ_id,
            bbox=geometry["bbox"],
            save_dir="./crop_image/totalseg_cropped_organ_imgs",
            filename=filename,
            margin_ratio=0.1,
            apply_mask=True,
            pad_to_square=True,
            pad_value=0,
        )

        feature_key = f"{filename}__organ_{organ_id}"

        database_index["slices"][filename]["organ_info"][str(organ_id)] = {
            "organ_name": organ_name,
            "organ_id": organ_id,

            "bbox": geometry["bbox"],
            "centroid": geometry["centroid"],
            "area": geometry["area"],
            "bbox_area": geometry["bbox_area"],

            "slice_relative_centroid": geometry["slice_relative_centroid"],
            "slice_relative_area": geometry["slice_relative_area"],
            "slice_relative_bbox_area": geometry["slice_relative_bbox_area"],

            "foreground_bbox": geometry["foreground_bbox"],
            "foreground_area": geometry["foreground_area"],
            "foreground_relative_centroid": geometry["foreground_relative_centroid"],
            "foreground_relative_area": geometry["foreground_relative_area"],
            "foreground_relative_bbox_area": geometry["foreground_relative_bbox_area"],

            "crop_path": crop_path,
            "lpips_feature_key": feature_key,
        }

with open("./results/totalseg_database_gt_index.json", "w") as f:
    json.dump(database_index, f, indent=2)
