from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from diffusion_sandbox.config import load_config
from diffusion_sandbox.seed import set_all_seeds
from diffusion_sandbox.logger import RunLogger
from diffusion_sandbox.data import build_dataloader
from diffusion_sandbox.image_data import build_cifar10_dataloader

from diffusion_sandbox.model import DDPM
from diffusion_sandbox.models import REGISTRY
from diffusion_sandbox.viz import scatter_2d, image_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()

def asdict_maybe(obj):
    if obj is None:
        return {}
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    if isinstance(obj, dict):
        return obj
    return {}

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Device and Seed
    if cfg.run.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.run.device)

    set_all_seeds(cfg.run.seed, cfg.run.cudnn.benchmark, cfg.run.cudnn.deterministic)

    # Data logger
    logger = RunLogger(cfg.run.output_dir, cfg.run.experiment_name, cfg.tracking.tensorboard, mlflow_cfg=cfg.tracking.mlflow.__dict__)
    logger.log_params({"config_path": args.config, **cfg.__dict__["run"].__dict__})

    # Data
    if cfg.data.name == "cifar10":
        dl = build_cifar10_dataloader(
            root=cfg.data.cfg["cifar10"]["root"],
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            download=cfg.data.cfg["cifar10"]["download"],
            class_conditional=cfg.data.cfg["cifar10"]["class_conditional"],
            img_size=cfg.data.cfg["cifar10"]["img_size"],
            cf_guidance_p=cfg.data.cfg["cifar10"]["cf_guidance_p"],
        )
        sample_shape = (cfg.train.sample_size, 3, cfg.data.cfg["cifar10"]["img_size"], cfg.data.cfg["cifar10"]["img_size"])
    else:
        dl = build_dataloader(
            name=cfg.data.name,
            n=cfg.data.num_samples,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            cfg_kwargs=cfg.data.cfg.get(cfg.data.name, {}),
            seed=cfg.run.seed,
        )
        sample_shape = None  # 点集默认 (n,2)
    # Model & Diffusion
    name = getattr(cfg.model, "name", "mlp_baseline")
    common = asdict_maybe(getattr(cfg.model, "common", None))
    specific = asdict_maybe(getattr(cfg.model, name, None))
    if name not in REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    ModelCls = REGISTRY[name]
    model = ModelCls(**common, **specific).to(device)

    ddpm = DDPM(
        timesteps=cfg.diffusion.timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        device=device,
    )

    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        for it, batch in enumerate(dl, start=1):
            if isinstance(batch, (tuple, list)):
                x, y = batch
                y = y.to(device)
            else:
                x, y = batch, None
            x = x.to(device)

            loss = ddpm.loss(model, x, y=y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            if global_step % cfg.train.log_interval == 0:
                logger.log_metric("train/loss", loss.item(), global_step)
            global_step += 1

        if epoch % cfg.train.sample_interval == 0 or epoch == cfg.train.epochs:
            with torch.no_grad():
                model.eval()
                if cfg.data.name == "cifar10":
                    fake = ddpm.sample(model, n=cfg.train.sample_size, sample_shape=sample_shape, y=None)
                    fig = image_grid(fake, nrow=8, title=f"epoch {epoch}")
                    logger.add_figure("samples/images", fig, step=epoch)
                else:
                    fake = ddpm.sample(model, n=cfg.train.sample_size)
                    real_batch = next(iter(dl))[: cfg.train.sample_size].to(device)
                    fig = scatter_2d(real_batch, fake, title=f"epoch {epoch}")
                    logger.add_figure("samples/compare", fig, step=epoch)

        if epoch % cfg.train.ckpt_interval == 0 or epoch == cfg.train.epochs:
            ckpt_path = Path(logger.run_dir) / f"model-epoch{epoch}.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "cfg": cfg.__dict__,
            }, ckpt_path)

    logger.close()


if __name__ == "__main__":
    main()
