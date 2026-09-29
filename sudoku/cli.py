"""Prepare data, train one frozen paper recipe, or evaluate its checkpoint."""

import argparse
import json
from pathlib import Path

from .data import prepare_data
from .evaluation import evaluate
from .training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="JUD hard-Sudoku experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--train-examples", type=int, default=48_000)
    prepare.add_argument("--validation-examples", type=int, default=2_000)

    training = commands.add_parser("train")
    training.add_argument("--objective", choices=("mse", "posterior_ce"), required=True)
    training.add_argument("--encoding", choices=("original", "shuffled"), default="original")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, default=101)
    training.add_argument("--steps", type=int, help="override the recipe length for a smoke test")
    training.add_argument("--global-batch", type=int, default=256)
    training.add_argument("--train-examples", type=int, default=48_000)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--checkpoint-every", type=int, default=5_000)
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--no-resume", action="store_true")

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--steps", type=int, default=180)
    evaluation.add_argument("--batch-size", type=int, default=64)
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--seed", type=int, default=2001)
    args = parser.parse_args()

    if args.command == "prepare-data":
        print(
            json.dumps(
                prepare_data(
                    args.output,
                    args.workers,
                    args.force,
                    args.train_examples,
                    args.validation_examples,
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        shuffled = args.encoding == "shuffled"
        train(
            TrainConfig(
                objective=args.objective,
                data=args.data,
                output=args.output,
                steps=args.steps if args.steps is not None else (20_000 if shuffled else 10_000),
                digit_permutation=shuffled,
                lr_schedule="flat" if shuffled else "cosine",
                min_learning_rate=0.0 if shuffled else 3e-5,
                lr_horizon=None if shuffled else 15_000,
                weight_decay=0.0 if shuffled else 0.05,
                seed=args.seed,
                global_batch=args.global_batch,
                train_examples=args.train_examples,
                workers=args.workers,
                checkpoint_every=args.checkpoint_every,
                log_every=args.log_every,
                resume=not args.no_resume,
            )
        )
    else:
        evaluate(
            args.checkpoint,
            args.data,
            args.output,
            args.steps,
            args.batch_size,
            args.limit,
            args.seed,
        )
