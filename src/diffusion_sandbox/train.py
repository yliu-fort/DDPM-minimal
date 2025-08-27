from __future__ import annotations
import argparse, os, signal
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from diffusion_sandbox.config import load_config, cfg_to_dict
from diffusion_sandbox.seed import set_all_seeds
from diffusion_sandbox.logger import RunLogger
from diffusion_sandbox.data import build_dataloader
from diffusion_sandbox.image_data import build_cifar10_dataloader, build_mnist_dataloader

from diffusion_sandbox.model import DDPM
from diffusion_sandbox.models import REGISTRY
from diffusion_sandbox.viz import scatter_2d, image_grid

from diffusion_sandbox.checkpointing import (
    save_checkpoint,
    maybe_resume,
    latest_checkpoint,
    prune_old,
)

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
            root=cfg.data.cfg[cfg.data.name]["root"],
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            download=cfg.data.cfg[cfg.data.name]["download"],
            class_conditional=cfg.data.cfg[cfg.data.name]["class_conditional"],
            img_size=cfg.data.cfg[cfg.data.name]["img_size"],
            cf_guidance_p=cfg.data.cfg[cfg.data.name]["cf_guidance_p"],
        )
        sample_shape = (cfg.train.sample_size, 3, cfg.data.cfg[cfg.data.name]["img_size"], cfg.data.cfg[cfg.data.name]["img_size"])
    elif cfg.data.name == "mnist":
        dl = build_mnist_dataloader(
            root=cfg.data.cfg[cfg.data.name]["root"],
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            download=cfg.data.cfg[cfg.data.name]["download"],
            class_conditional=cfg.data.cfg[cfg.data.name]["class_conditional"],
            img_size=cfg.data.cfg[cfg.data.name]["img_size"],
            cf_guidance_p=cfg.data.cfg[cfg.data.name]["cf_guidance_p"],
        )
        sample_shape = (cfg.train.sample_size, 1, cfg.data.cfg[cfg.data.name]["img_size"], cfg.data.cfg[cfg.data.name]["img_size"])
    else:
        dl = build_dataloader(
            name=cfg.data.name,
            n=cfg.data.num_samples,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            cfg_kwargs=cfg.data.cfg.get(cfg.data.name, {}),
            seed=cfg.run.seed,
        )
        sample_shape = None  # Pointset default shape (n,2)
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

    # ---------- checkpoint dir & resume ----------
    run_dir = Path(logger.run_dir)
    ckpt_dir = Path(getattr(cfg.train, "checkpoint_dir", "") or (run_dir / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step, start_epoch, _ = maybe_resume(
        resume_from=getattr(cfg.train, "resume_from", ""),
        checkpoint_dir=str(ckpt_dir),
        model=model,
        optimizer=opt,
        scheduler=None,
        scaler=None,
        ema=None,
        strict=True,
        restore_rng=getattr(cfg.train, "save_rng_state", True),
    )
    global_step = start_step

    # handle SIGINT/SIGTERM → save & exit cleanly
    stop_flag = {"stop": False}
    def _handle(sig, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
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

            # ---------- autosave by step ----------
            save_every = int(getattr(cfg.train, "save_every_steps", 0))
            if save_every and (global_step % save_every == 0 or stop_flag["stop"]):
                step_path = ckpt_dir / f"step_{global_step}.pt"
                save_checkpoint(
                    path=str(step_path),
                    model=model,
                    optimizer=opt,
                    scheduler=None,
                    scaler=None,
                    ema=None,
                    step=global_step,
                    epoch=epoch,
                    config=cfg_to_dict(cfg),
                    metrics=None,
                    save_rng_state=getattr(cfg.train, "save_rng_state", True),
                )
                # refresh latest.pt (hardlink, fallback copy)
                try:
                    latest = ckpt_dir / "latest.pt"
                    if latest.exists(): latest.unlink()
                    os.link(step_path, latest)
                except Exception:
                    import shutil as _sh
                    _sh.copy2(step_path, ckpt_dir / "latest.pt")
                prune_old(str(ckpt_dir), int(getattr(cfg.train, "keep_last_k", 3)))
                if stop_flag["stop"]:
                    print("[signal] checkpoint saved; exiting.")
                    logger.close()
                    return

        if epoch % cfg.train.sample_interval == 0 or epoch == cfg.train.epochs:
            with torch.no_grad():
                model.eval()
                if cfg.data.name in ["cifar10", "mnist"]:
                    fake = ddpm.sample(model, n=cfg.train.sample_size, sample_shape=sample_shape, y=None)
                    fig = image_grid(fake, nrow=8, title=f"epoch {epoch}")
                    logger.add_figure("samples/images", fig, step=epoch)
                else:
                    fake = ddpm.sample(model, n=cfg.train.sample_size)
                    real_batch = next(iter(dl))[: cfg.train.sample_size].to(device)
                    fig = scatter_2d(real_batch, fake, title=f"epoch {epoch}")
                    logger.add_figure("samples/compare", fig, step=epoch)

        if epoch % cfg.train.ckpt_interval == 0 or epoch == cfg.train.epochs:
            # keep your epoch checkpoint, but include optimizer/step/rng to make it resumable too
            ckpt_path = Path(logger.run_dir) / f"model-epoch{epoch}.pt"
            save_checkpoint(
                path=str(ckpt_path),
                model=model,
                optimizer=opt,
                scheduler=None,
                scaler=None,
                ema=None,
                step=global_step,
                epoch=epoch,
                config=cfg_to_dict(cfg),
                metrics=None,
                save_rng_state=getattr(cfg.train, "save_rng_state", True),
            )

    logger.close()


if __name__ == "__main__":
    main()
