import torch


def filter_valid_anatomical_labels(label_ids, num_labels):
    return label_ids[
        (label_ids >= 1) & (label_ids <= num_labels)
    ].long()


def build_anatomical_presence_mask(flat_seg, num_labels):
    """Return (B, O) mask indicating structures present in each image."""
    batch_size = flat_seg.shape[0]

    presence_mask = torch.zeros(
        batch_size,
        num_labels,
        device=flat_seg.device,
        dtype=torch.bool,
    )

    for i in range(batch_size):
        label_ids = filter_valid_anatomical_labels(
            torch.unique(flat_seg[i]),
            num_labels,
        )

        if label_ids.numel() > 0:
            presence_mask[i, label_ids - 1] = True

    return presence_mask


def build_region_overlap_mask(flat_seg, flat_region_mask, num_labels):
    """Return (B, O) mask indicating structures overlapping the query region."""
    batch_size = flat_seg.shape[0]

    overlap_mask = torch.zeros(
        batch_size,
        num_labels,
        device=flat_seg.device,
        dtype=torch.bool,
    )

    for i in range(batch_size):
        label_ids = filter_valid_anatomical_labels(
            torch.unique(flat_seg[i][flat_region_mask[i].bool()]),
            num_labels,
        )

        if label_ids.numel() > 0:
            overlap_mask[i, label_ids - 1] = True

    return overlap_mask
