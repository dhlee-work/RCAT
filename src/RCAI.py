import torch
import torch.nn as nn


class RegionConditionedPatchAdapter(nn.Module):
    """Inject a binary/soft region condition into patch embeddings."""

    def __init__(self, embed_dim, adapter_ratio=0.25, dropout=0.1):
        super().__init__()

        hidden_dim = int(embed_dim * adapter_ratio)

        self.norm = nn.LayerNorm(embed_dim)
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _prepare_region_condition(patch_embs, region_condition):
        batch_size, num_patches, _ = patch_embs.shape

        if region_condition is None:
            region_condition = torch.ones(
                batch_size,
                num_patches,
                dtype=patch_embs.dtype,
                device=patch_embs.device,
            )
        else:
            region_condition = region_condition.to(
                device=patch_embs.device,
                dtype=patch_embs.dtype,
            )

            if region_condition.dim() != 2:
                raise ValueError(
                    "region_condition must have shape (B, P), "
                    f"but got {tuple(region_condition.shape)}"
                )

            if region_condition.shape != (batch_size, num_patches):
                raise ValueError(
                    "region_condition shape mismatch: expected "
                    f"({batch_size}, {num_patches}), got "
                    f"{tuple(region_condition.shape)}"
                )

            empty_region = region_condition.sum(dim=1, keepdim=True) == 0
            if empty_region.any():
                region_condition = torch.where(
                    empty_region,
                    torch.ones_like(region_condition),
                    region_condition,
                )

        return region_condition.unsqueeze(-1)

    def forward(self, patch_embs, region_condition):
        region_condition = self._prepare_region_condition(
            patch_embs,
            region_condition,
        )

        patch_norm = self.norm(patch_embs)
        adapter_input = torch.cat([patch_norm, region_condition], dim=-1)
        patch_delta = self.adapter(adapter_input)
        patch_embs = self.out_norm(patch_embs + patch_delta)

        return patch_embs, region_condition


class PatchSelfAttention(nn.Module):
    """Self-attention over patch embeddings."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, average_attn_weights=False):
        super().__init__()

        self.average_attn_weights = average_attn_weights
        self.norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, patch_embs, need_weights=False):
        patch_norm = self.norm(patch_embs)
        attn_out, attn_weights = self.self_attn(
            query=patch_norm,
            key=patch_norm,
            value=patch_norm,
            need_weights=need_weights,
            average_attn_weights=self.average_attn_weights,
        )

        patch_embs = patch_embs + attn_out
        patch_embs = patch_embs + self.mlp(self.mlp_norm(patch_embs))

        return patch_embs, attn_weights if need_weights else None


class PatchGuidedTokenAttention(nn.Module):
    """Anatomical tokens attend to region-conditioned patch embeddings."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, average_attn_weights=False):
        super().__init__()

        self.average_attn_weights = average_attn_weights
        self.token_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, anatomical_tokens, patch_embs, need_weights=False):
        token_norm = self.token_norm(anatomical_tokens)
        patch_norm = self.patch_norm(patch_embs)

        attn_out, attn_weights = self.cross_attn(
            query=token_norm,
            key=patch_norm,
            value=patch_norm,
            need_weights=need_weights,
            average_attn_weights=self.average_attn_weights,
        )

        anatomical_tokens = anatomical_tokens + attn_out
        anatomical_tokens = anatomical_tokens + self.mlp(
            self.mlp_norm(anatomical_tokens)
        )

        return anatomical_tokens, attn_weights if need_weights else None


class AnatomicalTokenSelfAttention(nn.Module):
    """Self-attention across anatomical tokens."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, average_attn_weights=False):
        super().__init__()

        self.average_attn_weights = average_attn_weights
        self.norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, anatomical_tokens, need_weights=False):
        token_norm = self.norm(anatomical_tokens)
        attn_out, attn_weights = self.self_attn(
            query=token_norm,
            key=token_norm,
            value=token_norm,
            need_weights=need_weights,
            average_attn_weights=self.average_attn_weights,
        )

        anatomical_tokens = anatomical_tokens + attn_out
        anatomical_tokens = anatomical_tokens + self.mlp(
            self.mlp_norm(anatomical_tokens)
        )

        return anatomical_tokens, attn_weights if need_weights else None


class RegionConditionedAnatomicalInteractionLayer(nn.Module):
    """One RCAI block used by RCAT."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.1,
        adapter_ratio=0.25,
        average_attn_weights=False,
    ):
        super().__init__()

        self.region_patch_adapter = RegionConditionedPatchAdapter(
            embed_dim=embed_dim,
            adapter_ratio=adapter_ratio,
            dropout=dropout,
        )
        self.patch_self_attention = PatchSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            average_attn_weights=average_attn_weights,
        )
        self.patch_guided_token_attention = PatchGuidedTokenAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            average_attn_weights=average_attn_weights,
        )
        self.anatomical_token_self_attention = AnatomicalTokenSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            average_attn_weights=average_attn_weights,
        )

    def forward(
        self,
        anatomical_tokens,
        patch_embs,
        region_condition,
        need_weights=False,
    ):
        attn_out = {}

        patch_embs, used_region_condition = self.region_patch_adapter(
            patch_embs=patch_embs,
            region_condition=region_condition,
        )

        patch_embs, patch_self_weights = self.patch_self_attention(
            patch_embs=patch_embs,
            need_weights=need_weights,
        )

        anatomical_tokens, patch_guided_weights = self.patch_guided_token_attention(
            anatomical_tokens=anatomical_tokens,
            patch_embs=patch_embs,
            need_weights=need_weights,
        )

        anatomical_tokens, token_self_weights = self.anatomical_token_self_attention(
            anatomical_tokens=anatomical_tokens,
            need_weights=need_weights,
        )

        if need_weights:
            attn_out = {
                "region_conditioned_patch_adapter": {
                    "region_condition": used_region_condition,
                },
                "patch_self_attention": {
                    "weights": patch_self_weights,
                },
                "patch_guided_token_attention": {
                    "weights": patch_guided_weights,
                },
                "anatomical_token_self_attention": {
                    "weights": token_self_weights,
                },
            }

        return anatomical_tokens, patch_embs, attn_out


class RegionConditionedAnatomicalInteraction(nn.Module):
    """Stacked Region-Conditioned Anatomical Interaction (RCAI) blocks."""

    def __init__(
        self,
        embed_dim,
        depth=4,
        num_heads=8,
        dropout=0.1,
        adapter_ratio=0.25,
        average_attn_weights=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                RegionConditionedAnatomicalInteractionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    adapter_ratio=adapter_ratio,
                    average_attn_weights=average_attn_weights,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        anatomical_tokens,
        patch_embs,
        region_condition,
        need_weights=False,
    ):
        all_attn_out = [] if need_weights else None

        for layer_idx, layer in enumerate(self.layers):
            anatomical_tokens, patch_embs, attn_out = layer(
                anatomical_tokens=anatomical_tokens,
                patch_embs=patch_embs,
                region_condition=region_condition,
                need_weights=need_weights,
            )

            if need_weights:
                all_attn_out.append({
                    "layer_idx": layer_idx,
                    **attn_out,
                })

        return anatomical_tokens, patch_embs, all_attn_out
