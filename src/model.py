import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel

from src.RCAI import RegionConditionedAnatomicalInteraction
from src.model_helper import (
    build_anatomical_presence_mask,
    build_region_overlap_mask,
    filter_valid_anatomical_labels,
)
from src.utils import ClassificationHead, ProjModel, WarmupStepLR


class RCAT(pl.LightningModule):
    """Region-Conditioned Anatomical Token framework."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        encoder_kwargs = {
            "output_attentions": self.hparams.out_attn,
        }
        if self.hparams.out_attn:
            encoder_kwargs["attn_implementation"] = "eager"

        self.img_encoder = AutoModel.from_pretrained(
            self.hparams.patch_encoder_path,
            **encoder_kwargs,
        )

        self.patch_projection = ProjModel(
            in_dim=self.hparams.encoder_dim,
            hidden_dim=self.hparams.feature_dim,
            out_dim=self.hparams.feature_dim,
            dropout=0.1,
        )

        self.anatomical_tokens = nn.Parameter(
            torch.empty(1, self.hparams.num_labels, self.hparams.feature_dim)
        )
        nn.init.trunc_normal_(self.anatomical_tokens, std=0.02)

        self.rcai = RegionConditionedAnatomicalInteraction(
            embed_dim=self.hparams.feature_dim,
            depth=getattr(self.hparams, "rcai_depth", 4),
            num_heads=getattr(self.hparams, "rcai_num_heads", 8),
            dropout=getattr(self.hparams, "rcai_dropout", 0.1),
            adapter_ratio=getattr(self.hparams, "region_adapter_ratio", 0.25),
            average_attn_weights=getattr(
                self.hparams,
                "average_attn_weights",
                False,
            ),
        )

        self.asr_head = ClassificationHead(
            in_dim=self.hparams.feature_dim,
            hidden_dim=self.hparams.feature_dim // 2,
            out_dim=3,
            dropout=0.1,
        )

        self.img_encoder.requires_grad_(not self.hparams.img_encoder_freeze)

    def log_learning_rates(self, mode="train"):
        optimizer = self.optimizers()
        lr_names = ["img_encoder", "patch_projection", "other"]

        for i, param_group in enumerate(optimizer.param_groups):
            name = lr_names[i] if i < len(lr_names) else f"group_{i}"
            self.log(
                f"{mode}_lr_{name}",
                param_group["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    def configure_optimizers(self):
        encoder_params = []
        projection_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "img_encoder" in name:
                encoder_params.append(param)
            elif "patch_projection" in name:
                projection_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": self.hparams.lr_vit,
                "weight_decay": self.hparams.weight_decay,
            })

        if projection_params:
            param_groups.append({
                "params": projection_params,
                "lr": self.hparams.lr_proj,
                "weight_decay": self.hparams.weight_decay,
            })

        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            })

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = WarmupStepLR(
            optimizer=optimizer,
            warmup_steps=self.hparams.scheduler_t_up,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_step_lr",
            },
        }

    @staticmethod
    def zero_loss_like(ref_tensor):
        return ref_tensor.new_tensor(0.0)

    def asr_loss(self, logits, targets, mode="train"):
        """
        ASR classes:
            0: anatomical structure absent
            1: present but not overlapping the region condition
            2: present and overlapping the region condition
        """
        class_weight = torch.tensor(
            [0.3, 1.0, 1.0],
            device=logits.device,
            dtype=logits.dtype,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, 3),
            targets.reshape(-1),
            weight=class_weight,
        )

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()

            class_acc = []
            for class_idx in range(3):
                class_mask = targets == class_idx
                correct = ((preds == class_idx) & class_mask).sum().float()
                class_acc.append(correct / class_mask.sum().clamp(min=1))

            class_ratio = [
                (targets == class_idx).float().mean()
                for class_idx in range(3)
            ]

        if not self.hparams.disable_logger:
            self.log(f"{mode}_asr_loss", loss)
            self.log(f"{mode}_asr_acc", acc)
            self.log(f"{mode}_asr_absent_acc", class_acc[0])
            self.log(f"{mode}_asr_present_nonoverlap_acc", class_acc[1])
            self.log(f"{mode}_asr_present_overlap_acc", class_acc[2])
            self.log(f"{mode}_asr_absent_ratio", class_ratio[0])
            self.log(f"{mode}_asr_present_nonoverlap_ratio", class_ratio[1])
            self.log(f"{mode}_asr_present_overlap_ratio", class_ratio[2])

        return loss

    def rtca_loss(self, token_embs, region_embs, mode="train"):
        """Region-Token Contrastive Alignment with in-batch negatives."""
        num_pairs = token_embs.shape[0]

        logits = F.cosine_similarity(
            token_embs[:, None, :],
            region_embs[None, :, :],
            dim=-1,
        )
        logits = logits / self.hparams.rtca_temperature

        positive_logits = logits.diag()
        loss = (-positive_logits + torch.logsumexp(logits, dim=-1)).mean()

        with torch.no_grad():
            labels = torch.arange(num_pairs, device=token_embs.device)
            ranking = logits.argsort(dim=-1, descending=True)
            positive_rank = (ranking == labels[:, None]).nonzero()[:, 1]

        if not self.hparams.disable_logger:
            self.log(f"{mode}_rtca_loss", loss)
            self.log(f"{mode}_rtca_acc_top1", (positive_rank == 0).float().mean())
            self.log(f"{mode}_rtca_acc_top5", (positive_rank < 5).float().mean())
            self.log(
                f"{mode}_rtca_mean_positive_rank",
                1 + positive_rank.float().mean(),
            )

        return loss

    def build_patch_ratio_from_binary_mask(
        self,
        binary_masks,
        image_size=None,
        patch_grid_size=None,
    ):
        if image_size is None:
            image_size = tuple(self.hparams.patch_encoder_size)

        if patch_grid_size is None:
            patch_grid_size = tuple(self.hparams.patch_grid_size)

        if binary_masks.dim() == 3:
            binary_masks = binary_masks.unsqueeze(1)

        binary_masks = binary_masks.float()

        if binary_masks.shape[-2:] != image_size:
            binary_masks = F.interpolate(
                binary_masks,
                size=image_size,
                mode="nearest",
            )

        grid_h, grid_w = patch_grid_size
        patch_h = image_size[0] // grid_h
        patch_w = image_size[1] // grid_w

        patch_ratio = F.avg_pool2d(
            binary_masks,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

        return patch_ratio.flatten(1)

    def resize_region_condition(self, masks):
        """Convert a pixel-level query mask into a patch-level condition."""
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        masks = masks.float()

        image_h, image_w = self.hparams.patch_encoder_size
        grid_h, grid_w = self.hparams.patch_grid_size
        patch_h = image_h // grid_h
        patch_w = image_w // grid_w

        mask_img = F.interpolate(
            masks,
            size=(image_h, image_w),
            mode="nearest",
        )

        patch_ratio = F.avg_pool2d(
            mask_img,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        ).flatten(1)

        threshold = getattr(self.hparams, "region_patch_threshold", 0)
        return (patch_ratio > threshold).to(dtype=torch.float32)


    def encode_retrieval_features(
        self,
        imgs,
        region_masks=None,
        need_weights=False,
    ):
        """Encode CT slices into RCAT anatomical tokens for retrieval.

        Args:
            imgs: Tensor [B, C, H, W].
            region_masks: Optional pixel-level region condition [B, 1, H, W]
                or [B, H, W]. If None, the full-image condition is used.
            need_weights: Whether RCAI attention weights should be returned.

        Returns:
            dict with projected patch embeddings, anatomical tokens, ASR logits,
            ASR probabilities, selectivity weights p(class=2), and attentions.
        """
        enc_out = self.img_encoder(imgs)
        patch_embs_enc = enc_out.last_hidden_state[:, 1:, :]
        projected_patch_embs = self.patch_projection(patch_embs_enc)

        if region_masks is None:
            region_condition = None
        else:
            region_condition = self.resize_region_condition(region_masks)

        batch_size = imgs.shape[0]
        anatomical_tokens = self.anatomical_tokens.expand(batch_size, -1, -1)

        anatomical_tokens, rcai_patch_embs, attentions = self.rcai(
            anatomical_tokens=anatomical_tokens,
            patch_embs=projected_patch_embs,
            region_condition=region_condition,
            need_weights=need_weights,
        )

        asr_logits = self.asr_head(anatomical_tokens)
        asr_probs = torch.softmax(asr_logits, dim=-1)
        selectivity_weights = asr_probs[..., 2]

        return {
            "projected_patch_embs": projected_patch_embs,
            "rcai_patch_embs": rcai_patch_embs,
            "anatomical_tokens": anatomical_tokens,
            "asr_logits": asr_logits,
            "asr_probs": asr_probs,
            "selectivity_weights": selectivity_weights,
            "attentions": attentions,
        }

    def run_step(self, batch, step_mode):
        imgs = batch["img"]
        segs = batch["seg"].to(imgs.device)
        region_masks = batch["mask"].to(imgs.device)

        batch_size = imgs.shape[0]
        device = imgs.device

        # --------------------------------------------------
        # 1-3. Encoder + RCAI + ASR
        # --------------------------------------------------
        encoded = self.encode_retrieval_features(
            imgs=imgs,
            region_masks=region_masks,
            need_weights=self.hparams.out_attn,
        )
        patch_embs = encoded["projected_patch_embs"]
        anatomical_tokens = encoded["anatomical_tokens"]
        asr_logits = encoded["asr_logits"]

        # --------------------------------------------------
        # 3. Anatomical Selectivity Recognition (ASR) targets
        # --------------------------------------------------
        flat_seg = segs.reshape(batch_size, -1)
        flat_region_mask = region_masks.reshape(batch_size, -1).bool()

        anatomical_presence = build_anatomical_presence_mask(
            flat_seg,
            self.hparams.num_labels,
        )
        region_overlap = build_region_overlap_mask(
            flat_seg,
            flat_region_mask,
            self.hparams.num_labels,
        )

        asr_targets = torch.zeros(
            batch_size,
            self.hparams.num_labels,
            device=device,
            dtype=torch.long,
        )
        asr_targets[anatomical_presence] = 1
        asr_targets[region_overlap] = 2

        asr_loss = self.asr_loss(
            logits=asr_logits,
            targets=asr_targets,
            mode=step_mode,
        )

        # --------------------------------------------------
        # 4. Region-Token Contrastive Alignment (RTCA)
        # --------------------------------------------------
        # The target region representation is constructed from the same
        # image using detached encoder-projected patch embeddings.
        # Only anatomical structures overlapping the sampled region
        # condition participate in RTCA.
        rtca_loss = self.zero_loss_like(patch_embs)

        if segs.dim() == 4:
            seg_maps = segs.squeeze(1).long()
        else:
            seg_maps = segs.long()

        if region_masks.dim() == 4:
            region_masks_bool = region_masks.squeeze(1).bool()
        else:
            region_masks_bool = region_masks.bool()

        target_patch_embs = patch_embs.detach()
        token_embs = []
        region_embs = []

        max_regions = getattr(
            self.hparams,
            "max_rtca_regions_per_image",
            10,
        )

        for b in range(batch_size):
            selected_labels = torch.unique(
                seg_maps[b][region_masks_bool[b]]
            ).long()
            selected_labels = filter_valid_anatomical_labels(
                selected_labels,
                self.hparams.num_labels,
            )

            if selected_labels.numel() == 0:
                continue

            if max_regions is not None and selected_labels.numel() > max_regions:
                selected_labels = selected_labels[
                    torch.randperm(
                        selected_labels.numel(),
                        device=selected_labels.device,
                    )[:max_regions]
                ]

            for label_id in selected_labels:
                label_id_int = int(label_id.item())
                region_pixel_mask = (
                    (seg_maps[b] == label_id_int)
                    & region_masks_bool[b]
                )

                patch_weight = self.build_patch_ratio_from_binary_mask(
                    region_pixel_mask.unsqueeze(0)
                )[0].to(
                    device=device,
                    dtype=target_patch_embs.dtype,
                )

                weight_sum = patch_weight.sum()
                if weight_sum.item() <= 0:
                    continue

                region_emb = (
                    target_patch_embs[b] * patch_weight[:, None]
                ).sum(dim=0) / weight_sum.clamp_min(1e-6)

                token_embs.append(anatomical_tokens[b, label_id_int - 1])
                region_embs.append(region_emb)

        if region_embs:
            token_embs = torch.stack(token_embs, dim=0)
            region_embs = torch.stack(region_embs, dim=0).detach()

            if not self.hparams.disable_logger:
                self.log(
                    f"{step_mode}_rtca_pair_count",
                    torch.tensor(float(region_embs.shape[0]), device=device),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

            rtca_loss = self.rtca_loss(
                token_embs=token_embs,
                region_embs=region_embs,
                mode=step_mode,
            )

        total_loss = (
            getattr(self.hparams, "asr_lambda", 1.0) * asr_loss
            + getattr(self.hparams, "rtca_lambda", 0.1) * rtca_loss
        )

        if not self.hparams.disable_logger:
            self.log(
                f"{step_mode}_loss",
                total_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            if step_mode == "train":
                self.log_learning_rates(mode=step_mode)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, "val")
