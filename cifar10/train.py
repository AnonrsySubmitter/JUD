# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from NVIDIA EDM (https://github.com/NVlabs/edm), CC BY-NC-SA 4.0; see cifar10/LICENSE-NVIDIA.txt.

"""Train the paper's JUD model or standard EDM control."""

import json
import os
import re

import click
import dnnlib
import torch
from torch_utils import distributed as dist
from training import training_loop


@click.command()
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
@click.option("--data", required=True, type=click.Path(exists=True))
@click.option("--precond", type=click.Choice(["jud", "edm"]), default="jud", show_default=True)
@click.option(
    "--duration",
    type=click.FloatRange(min=0, min_open=True),
    default=200,
    show_default=True,
    help="Total training budget in million images, including a resumed pilot.",
)
@click.option("--batch", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--batch-gpu", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--bf16", type=bool, default=None, help="Use BF16 (default: JUD yes, EDM no).")
@click.option("--workers", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--jud-time-sampling",
    type=click.Choice(["uniform", "cdf"]),
    default="uniform",
    show_default=True,
)
@click.option("--jud-time-cdf", type=click.Path(exists=True, dir_okay=False))
@click.option("--resume", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--tick",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Logging interval in thousand images.",
)
@click.option(
    "--snap",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Snapshot interval in ticks; always save at completion.",
)
@click.option(
    "--dump",
    type=click.IntRange(min=1),
    default=500,
    show_default=True,
    help="Training-state interval in ticks; always save at completion.",
)
@click.option("--dry-run", is_flag=True)
def main(**kwargs):
    """Train DDPM++ with the paper's architecture, optimizer and augmentation."""
    opts = dnnlib.EasyDict(kwargs)
    if opts.precond == "edm" and opts.jud_time_sampling != "uniform":
        raise click.ClickException("CDF time sampling is specific to JUD")
    if opts.jud_time_sampling == "cdf" and opts.jud_time_cdf is None:
        raise click.ClickException("CDF sampling requires --jud-time-cdf")
    if os.path.exists(opts.outdir) and not opts.dry_run:
        raise click.ClickException("--outdir must be a new directory")

    c = dnnlib.EasyDict(
        run_dir=opts.outdir,
        dataset_kwargs={
            "class_name": "training.dataset.ImageFolderDataset",
            "path": opts.data,
            "use_labels": False,
            "xflip": False,
            "cache": True,
        },
        data_loader_kwargs={"pin_memory": True, "num_workers": opts.workers, "prefetch_factor": 2},
        network_kwargs={
            "class_name": "training.networks.JumpUpDownPrecond"
            if opts.precond == "jud"
            else "training.networks.EDMPrecond",
            "model_type": "SongUNet",
            "embedding_type": "positional",
            "encoder_type": "standard",
            "decoder_type": "standard",
            "channel_mult_noise": 1,
            "resample_filter": [1, 1],
            "model_channels": 128,
            "channel_mult": [2, 2, 2],
            "augment_dim": 9,
            "dropout": 0.13,
            "use_fp16": False,
        },
        loss_kwargs={
            "class_name": "training.loss.JumpUpDownLoss"
            if opts.precond == "jud"
            else "training.loss.EDMLoss"
        },
        optimizer_kwargs={
            "class_name": "torch.optim.Adam",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        augment_kwargs={
            "class_name": "training.augment.AugmentPipe",
            "p": 0.12,
            "xflip": 1e8,
            "yflip": 1,
            "scale": 1,
            "rotate_frac": 1,
            "aniso": 1,
            "translate_frac": 1,
        },
        total_kimg=max(int(opts.duration * 1000), 1),
        batch_size=opts.batch,
        batch_gpu=opts.batch_gpu,
        ema_halflife_kimg=500,
        ema_rampup_ratio=0.05,
        lr_rampup_kimg=10000,
        loss_scaling=1,
        grad_clip=0,
        cudnn_benchmark=True,
        kimg_per_tick=opts.tick,
        snapshot_ticks=opts.snap,
        state_dump_ticks=opts.dump,
        seed=opts.seed,
    )
    if opts.bf16 if opts.bf16 is not None else opts.precond == "jud":
        c.network_kwargs["amp_dtype"] = "bf16"
    if opts.precond == "jud":
        c.loss_kwargs.update(time_sampling=opts.jud_time_sampling, time_cdf_path=opts.jud_time_cdf)
    dataset = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    if dataset.resolution != 32 or dataset.num_channels != 3:
        raise click.ClickException("Expected RGB CIFAR-10 images at 32×32")
    c.dataset_kwargs.update(resolution=32, max_size=len(dataset))
    dataset.close()

    if opts.resume is not None:
        match = re.fullmatch(r"training-state-(\d+)\.pt", os.path.basename(opts.resume))
        if match is None:
            raise click.ClickException("--resume must name training-state-XXXXXX.pt")
        c.resume_state_dump = opts.resume
        c.resume_kimg = int(match.group(1))
        c.resume_pkl = os.path.join(
            os.path.dirname(opts.resume), f"network-snapshot-{match.group(1)}.pkl"
        )
        if not os.path.isfile(c.resume_pkl):
            raise click.ClickException("Resume requires the matching network snapshot")
        if c.total_kimg <= c.resume_kimg:
            raise click.ClickException("--duration must exceed the resumed progress")
    if opts.dry_run:
        print(json.dumps(c, indent=2))
        return

    torch.multiprocessing.set_start_method("spawn")
    dist.init()
    if opts.batch % dist.get_world_size():
        raise click.ClickException("--batch must be divisible by the number of GPUs")
    per_gpu = opts.batch // dist.get_world_size()
    if per_gpu % min(opts.batch_gpu, per_gpu):
        raise click.ClickException("--batch-gpu must divide the per-GPU batch")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir)
        with open(os.path.join(c.run_dir, "training_options.json"), "w") as file:
            json.dump(c, file, indent=2)
    dist.print0(json.dumps(c, indent=2))
    training_loop.training_loop(**c)


if __name__ == "__main__":
    main()
