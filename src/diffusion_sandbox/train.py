from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from diffusion_sandbox.config import load_config
from diffusion_sandbox.seed import set_all_seeds
from diffusion_sandbox.logger import RunLogger
from diffusion_sandbox.data import build_dataloader, SyntheticGMMCfg, SyntheticRingCfg
from diffusion_sandbox.model import MLPNoisePredictor, DDPM
from diffusion_sandbox.viz import scatter_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # 设备与种子
    if cfg.run.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.run.device)

    set_all_seeds(cfg.run.seed, cfg.run.cudnn.benchmark, cfg.run.cudnn.deterministic)

    # 日志器
    logger = RunLogger(cfg.run.output_dir, cfg.run.experiment_name, cfg.tracking.tensorboard, mlflow_cfg=cfg.tracking.mlflow.__dict__)
    logger.log_params({"config_path": args.config, **cfg.__dict__["run"].__dict__})

    # 数据
    dl = build_dataloader(
        name=cfg.data.name,
        n=cfg.data.num_samples,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        gmm=SyntheticGMMCfg(**cfg.data.gmm.__dict__),
        ring=SyntheticRingCfg(**cfg.data.ring.__dict__),
        seed=cfg.run.seed,
    )

    # 模型 & 扩散
    model = MLPNoisePredictor(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        time_embed_dim=cfg.model.time_embed_dim,
    ).to(device)

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
        for it, x in enumerate(dl, start=1):
            x = x.to(device)
            loss = ddpm.loss(model, x)
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
                fake = ddpm.sample(model, cfg.train.sample_size)
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
