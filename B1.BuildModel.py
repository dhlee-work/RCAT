import argparse
import glob
import multiprocessing
import os
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.datautils import RCATDataset
from src.model import RCAT


def parse_args():
    parser = argparse.ArgumentParser(description="Train RCAT.")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/Model-Totalseg-RCAT.yaml",
        help="Path to the RCAT configuration file.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Optional RCAT checkpoint for resuming training.",
    )
    return parser.parse_args()


def build_logger_and_callbacks(config):
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path("logs") / config.project_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, run_dir / "config.yaml")

    logger = None
    if not config.disable_logger:
        logger_kwargs = {
            "name": run_id,
            "project": config.project_name,
            "log_model": False,
            "save_dir": str(run_dir),
        }
        if getattr(config, "logger_id", None):
            logger_kwargs.update(
                id=config.logger_id,
                resume="allow",
            )
        logger = WandbLogger(**logger_kwargs)

    checkpoint_every_n_steps = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="step-{step:07d}",
        every_n_train_steps=config.ckpt_every_n_steps,
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    best_checkpoint = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-{step:07d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_every_n_steps, best_checkpoint]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    return logger, callbacks, run_dir


def build_dataloader(dataset, config, shuffle, drop_last):
    kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": True,
    }

    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    pl.seed_everything(config.seed_number, workers=True)

    train_paths = sorted(
        glob.glob(os.path.join(config.root_dir, "train", "image", "*.npy"))
    )
    val_paths = sorted(
        glob.glob(os.path.join(config.root_dir, "gallery", "image", "*.npy"))
    )

    if not train_paths:
        raise FileNotFoundError("No training images were found.")
    if not val_paths:
        raise FileNotFoundError("No validation/gallery images were found.")

    print(f"Training slices: {len(train_paths):,}")
    print(f"Validation slices: {len(val_paths):,}")

    train_dataset = RCATDataset(train_paths, config=config)
    val_dataset = RCATDataset(val_paths, config=config)

    train_loader = build_dataloader(
        train_dataset,
        config=config,
        shuffle=True,
        drop_last=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        config=config,
        shuffle=False,
        drop_last=False,
    )

    logger, callbacks, run_dir = build_logger_and_callbacks(config)
    print(f"Run directory: {run_dir}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.device,
        max_steps=config.max_steps,
        max_epochs=-1,
        val_check_interval=config.val_check_interval,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed",
    )

    model = RCAT(config)

    if args.resume_checkpoint is not None:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume_checkpoint}"
            )
        print(f"Resuming from: {args.resume_checkpoint}")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_checkpoint,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
