from contextlib import nullcontext
from pathlib import Path

import torch

from src.model import RCAT


def resolve_checkpoint_path(project_name, checkpoint=None, logs_root="./logs"):
    """Resolve an explicit checkpoint or the most recently modified last.ckpt."""
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    project_dir = Path(logs_root) / project_name
    candidates = list(project_dir.glob("*/last.ckpt"))

    if not candidates:
        candidates = list(project_dir.glob("*/*.ckpt"))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {project_dir}. "
            "Pass --checkpoint explicitly."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _legacy_to_rcat_key(key):
    """Map the final legacy RBMIR/ROA checkpoint names to RCAT names."""
    replacements = [
        ("patch_embs_proj.", "patch_projection."),
        ("organ_tokens", "anatomical_tokens"),
        ("OVR_module.", "asr_head."),
        ("ROA.", "rcai."),
        (".patch_adapter.", ".region_patch_adapter."),
        (".patch_self_attn.", ".patch_self_attention."),
        (".iga.", ".patch_guided_token_attention."),
        (".osa.", ".anatomical_token_self_attention."),
        (".patch_self_attention.norm_p.", ".patch_self_attention.norm."),
        (".patch_self_attention.norm_mlp.", ".patch_self_attention.mlp_norm."),
        (
            ".patch_guided_token_attention.norm_q.",
            ".patch_guided_token_attention.token_norm.",
        ),
        (
            ".patch_guided_token_attention.norm_p.",
            ".patch_guided_token_attention.patch_norm.",
        ),
        (
            ".patch_guided_token_attention.norm_mlp.",
            ".patch_guided_token_attention.mlp_norm.",
        ),
        (
            ".anatomical_token_self_attention.norm_q.",
            ".anatomical_token_self_attention.norm.",
        ),
        (
            ".anatomical_token_self_attention.norm_mlp.",
            ".anatomical_token_self_attention.mlp_norm.",
        ),
    ]

    for old, new in replacements:
        key = key.replace(old, new)
    return key


def convert_legacy_state_dict(state_dict):
    return {
        _legacy_to_rcat_key(key): value
        for key, value in state_dict.items()
    }


def load_rcat_for_inference(checkpoint_path, config, device):
    """Load either a refactored RCAT checkpoint or the final legacy checkpoint.

    The compatibility path changes state-dict names only; tensor values are not
    modified.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = RCAT(config)

    is_legacy = any(
        key.startswith("ROA.")
        or key.startswith("OVR_module.")
        or key.startswith("patch_embs_proj.")
        or key == "organ_tokens"
        for key in state_dict.keys()
    )

    if is_legacy:
        state_dict = convert_legacy_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with RCAT after name conversion.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    model = model.to(device)
    model.eval()
    return model, is_legacy


def inference_autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def resolve_device(config, device=None):
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        configured_devices = getattr(config, "device", None)
        if configured_devices:
            return torch.device(f"cuda:{int(configured_devices[0])}")
        return torch.device("cuda")

    return torch.device("cpu")
