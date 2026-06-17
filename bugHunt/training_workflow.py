"""Operator workflow for cleanup, live training, and reinforcement rounds."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from paths import LOGS_DIR, MODELS_DIR


STALE_LOG_NAMES = [
    "advanced_train_live.pid",
    "nohup_probe.log",
    "live_collect_retry.log",
    "advanced_train_live_20260613T131928Z.log",
    "advanced_train_live_20260613T132145Z.log",
]
STALE_LOG_GLOBS = ["*.log.path", "*.partial.jsonl"]


def cleanup_stale_artifacts() -> None:
    """Remove stale local run files while preserving promoted models/checkpoints."""
    removed = 0
    for pattern in STALE_LOG_GLOBS:
        for path in LOGS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    for name in STALE_LOG_NAMES:
        path = LOGS_DIR / name
        if path.exists():
            path.unlink()
            removed += 1
    for cache_dir in PROJECT_DIR.rglob("__pycache__"):
        shutil.rmtree(cache_dir)
        removed += 1
    for pyc in PROJECT_DIR.rglob("*.pyc"):
        pyc.unlink()
        removed += 1
    print(f"[WORKFLOW] cleanup removed {removed} stale artifact(s)")
    print(f"[WORKFLOW] keeping best model: {MODELS_DIR / 'advanced_model_best.pt'}")


def print_options() -> None:
    """Show the supported run modes and profile intent."""
    print("[WORKFLOW] modes:")
    print("[WORKFLOW]   options   show this menu and exit")
    print("[WORKFLOW]   new       train one fresh model")
    print("[WORKFLOW]   reinforce reinforce the current models/advanced_model_best.pt")
    print("[WORKFLOW]   cycle     train fresh once, then reinforce for remaining rounds")
    print("[WORKFLOW] data:")
    print("[WORKFLOW]   --data-percent N selects how much of the large bulk caches to stream")
    print("[WORKFLOW]   curated raw files are always loaded; bulk caches are 0% unless selected")
    print("[WORKFLOW] examples:")
    print("[WORKFLOW]   python bugHunt/training_workflow.py --mode options")
    print("[WORKFLOW]   python bugHunt/training_workflow.py --mode new --collect cached --data-percent 1")
    print("[WORKFLOW]   python bugHunt/training_workflow.py --mode cycle --collect cached --data-percent 10 --rounds 3")
    print("[WORKFLOW] append trainer overrides after --, for example: -- --epochs 10 --chain-weight 1.5")


def extra_has(extra_args: list[str], option: str) -> bool:
    return option in extra_args or any(arg.startswith(f"{option}=") for arg in extra_args)


def round_needs_resume(mode: str, round_index: int) -> bool:
    return mode == "reinforce" or (mode == "cycle" and round_index > 0)


def build_train_command(args, round_index: int, total_rounds: int,
                        extra_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "advanced_train.py"),
        "--training-profile", args.training_profile,
        "--print-options",
    ]
    if args.collect == "cached":
        cmd.append("--skip-collect")
    if not args.no_update_default_best:
        cmd.append("--update-default-best")
    if args.no_periodic_checkpoints:
        cmd.append("--no-periodic-checkpoints")
    if args.require_cuda:
        cmd.append("--require-cuda")
    if args.dry_run_trainer:
        cmd.append("--dry-run")
    if args.data_percent is not None and not extra_has(extra_args, "--data-percent"):
        cmd.extend(["--data-percent", str(args.data_percent)])
    if round_needs_resume(args.mode, round_index):
        cmd.append("--resume-latest-best")
    if args.run_name_prefix and not extra_has(extra_args, "--run-name"):
        cmd.extend(["--run-name", f"{args.run_name_prefix}_r{round_index + 1:02d}"])
    cmd.extend(extra_args)
    print(f"[WORKFLOW] round {round_index + 1}/{total_rounds}: {shlex.join(cmd)}")
    return cmd


def run_training_rounds(args, extra_args: list[str]) -> None:
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    total_rounds = args.rounds if args.mode in {"reinforce", "cycle"} else 1
    for round_index in range(total_rounds):
        cmd = build_train_command(args, round_index, total_rounds, extra_args)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=PROJECT_DIR)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Hunter Trainer cleanup/new-training/reinforcement workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["options", "new", "reinforce", "cycle"], default="options")
    parser.add_argument("--training-profile", choices=["min", "default", "max"], default="default")
    parser.add_argument("--rounds", type=int, default=1, help="Reinforcement/cycle rounds; no hard upper cap")
    parser.add_argument("--collect", choices=["live", "cached"], default="live")
    parser.add_argument("--data-percent", type=float, default=None,
                        help="Percent of large bulk caches to stream during each round")
    parser.add_argument("--clean", action="store_true", help="Remove stale local artifacts before running")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching trainer")
    parser.add_argument("--dry-run-trainer", action="store_true", help="Launch trainer in its own dry-run mode")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-periodic-checkpoints", action="store_true")
    parser.add_argument("--no-update-default-best", action="store_true")
    parser.add_argument("--run-name-prefix", default=None)
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if args.clean:
        cleanup_stale_artifacts()
    print_options()
    if args.mode == "options":
        return
    run_training_rounds(args, extra_args)


if __name__ == "__main__":
    main()
