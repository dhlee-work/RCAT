import json
import os
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoImageProcessor

from src.datautils_helper import (
    apply_label_mapping,
    make_region_mask_from_ids,
    normalize_ct,
    pad_to_square,
    random_region_ids,
)
from src.utils import parse_gallery_query_key


class _RCATBaseDataset(Dataset):
    """Shared CT preprocessing used by RCAT datasets."""

    def __init__(self, config):
        self.config = config
        self.patch_enc_processor = AutoImageProcessor.from_pretrained(
            self.config.patch_encoder_path
        )
        self.patch_enc_processor.size["shortest_edge"] = (
            self.config.patch_encoder_size[0]
        )
        self.patch_enc_processor.do_center_crop = False

        self.resize_seg = transforms.Resize(
            self.config.gt_seg_size,
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )

        self.label_index_converter = None

    def _load_image(self, image_path):
        slice_img = np.load(image_path, allow_pickle=True)
        slice_img = normalize_ct(slice_img)
        slice_img = pad_to_square(slice_img)

        slice_img_pil = Image.fromarray(slice_img, mode="L").convert("RGB")
        slice_img_pil = slice_img_pil.resize(
            tuple(self.config.patch_encoder_size),
            Image.BICUBIC,
        )

        return self.patch_enc_processor(
            images=slice_img_pil,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

    def _segmentation_path(self, image_path):
        image_path = str(image_path)
        seg_dir = "seg_vista3d" if self.config.vista3d else "seg_gt"
        seg_path = image_path.replace("/image/", f"/{seg_dir}/")
        return str(Path(seg_path).with_suffix(".npy"))

    def _load_segmentation(self, image_path):
        seg_path = self._segmentation_path(image_path)
        slice_seg = np.load(seg_path, allow_pickle=True)

        if self.config.vista3d:
            if self.label_index_converter is None:
                converter_path = getattr(
                    self.config,
                    "vista3d_converter_path",
                    "./datasets/vista3d_index_converter.json",
                )
                with open(converter_path, "r") as f:
                    self.label_index_converter = json.load(f)

            slice_seg = apply_label_mapping(
                slice_seg,
                self.label_index_converter["vista3d2model"],
            )

        slice_seg = pad_to_square(slice_seg)
        seg = torch.as_tensor(slice_seg).long()
        return self.resize_seg(seg.unsqueeze(0).float()).long()


class RCATDataset(_RCATBaseDataset):
    """Training/validation dataset with sampled region conditions."""

    def __init__(self, image_paths, config=None):
        super().__init__(config=config)
        self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = self._load_image(image_path)
        seg = self._load_segmentation(image_path)

        if np.random.uniform(0.0, 1.0) <= self.config.region_sampling_rate:
            selected_region_ids = random_region_ids(seg)
            region_mask = (
                make_region_mask_from_ids(seg, selected_region_ids)
                if selected_region_ids
                else torch.ones_like(seg)
            )
        else:
            region_mask = torch.ones_like(seg)

        return {
            "img": img,
            "seg": seg,
            "mask": region_mask,
            "filename": str(image_path),
        }


class RCATImageDataset(_RCATBaseDataset):
    """Image-only dataset for database construction.

    No segmentation is read. Database tokens are therefore constructed with
    the full-image region condition without region annotations or an external
    segmentation model.
    """

    def __init__(self, image_paths, config=None):
        super().__init__(config=config)
        self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        return {
            "img": self._load_image(image_path),
            "filename": str(image_path),
        }


class RCATQueryDataset(_RCATBaseDataset):
    """Dataset for predefined single- or multi-region retrieval queries."""

    def __init__(
        self,
        selected_queries,
        config=None,
        split="gallery",
    ):
        super().__init__(config=config)
        self.selected_queries = selected_queries
        self.query_keys = list(selected_queries.keys())
        self.split = split

    def __len__(self):
        return len(self.query_keys)

    def _image_path_from_query_key(self, query_key):
        slice_name, _ = parse_gallery_query_key(query_key)
        return os.path.join(
            self.config.root_dir,
            self.split,
            "image",
            f"{slice_name}.npy",
        )

    def __getitem__(self, idx):
        query_key = self.query_keys[idx]
        image_path = self._image_path_from_query_key(query_key)
        _, region_ids_from_key = parse_gallery_query_key(query_key)

        value = self.selected_queries[query_key]
        if isinstance(value, dict):
            selected_region_ids = [
                int(x) for x in value.get("query_idx", region_ids_from_key)
            ]
        else:
            selected_region_ids = [int(x) for x in value]

        img = self._load_image(image_path)

        if selected_region_ids == [-1]:
            region_mask = torch.ones(
                (1, *self.config.gt_seg_size),
                dtype=torch.long,
            )
        else:
            seg = self._load_segmentation(image_path)
            ids_tensor = torch.tensor(
                selected_region_ids,
                dtype=seg.dtype,
            )
            region_mask = torch.isin(seg, ids_tensor).long()

            if region_mask.sum() == 0:
                raise ValueError(
                    f"Query region is empty for {query_key}: "
                    f"selected_region_ids={selected_region_ids}"
                )

        return {
            "img": img,
            "mask": region_mask,
            "filename": str(image_path),
            "query_key": str(query_key),
        }
