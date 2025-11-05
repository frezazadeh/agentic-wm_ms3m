"""
Entry point for training the WM–MS³M PRB-conditioned world model.

Usage (from repo root):
    python scripts/train_wm_ms3m.py --data-dir ./data --ckpt-dir ./artifacts
"""

import argparse
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SRC_DIR = os.path.join(ROOT_DIR, "src")

for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from wm_ms3m import WMMS3MConfig, train_and_eval_wm_ms3m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing x.npy and y.npy (default: use Config.data_dir)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory to store checkpoints & metrics (default: use Config.ckpt_dir)",
    )
    args = parser.parse_args()

    cfg = WMMS3MConfig()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir

    _ = train_and_eval_wm_ms3m(cfg)


if __name__ == "__main__":
    main()
