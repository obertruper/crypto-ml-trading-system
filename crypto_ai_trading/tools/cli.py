#!/usr/bin/env python3
"""
Unified CLI wrapper for common tasks: eval and monitor.
Usage:
  python -m tools.cli eval [--checkpoint PATH]
  python -m tools.cli monitor
"""
import argparse
import subprocess
from pathlib import Path
import sys

def run_eval(checkpoint: str | None):
    # Prefer evaluate_model_simple.py; it auto-detects latest best_*.pth if no checkpoint passed
    cmd = [sys.executable, "evaluate_model_simple.py"]
    if checkpoint:
        # The simple script doesn't accept args; ensure file exists and print hint
        if not Path(checkpoint).exists():
            print(f"❌ Checkpoint not found: {checkpoint}")
            sys.exit(1)
        print("ℹ️ evaluate_model_simple.py не принимает аргументы; использует последний best_model_*.pth.")
        print(f"   Указанный путь можно переименовать в формат best_model_*.pth: {checkpoint}")
    subprocess.run(cmd, check=False)

def run_monitor():
    cmd = [sys.executable, "monitor_training.py"]
    subprocess.run(cmd, check=False)

def main():
    parser = argparse.ArgumentParser(description="Unified CLI for evaluation and monitoring")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run simple evaluation")
    p_eval.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")

    sub.add_parser("monitor", help="Run training monitor")

    args = parser.parse_args()

    if args.command == "eval":
        run_eval(args.checkpoint)
    elif args.command == "monitor":
        run_monitor()

if __name__ == "__main__":
    main()
