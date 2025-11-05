"""
Entry point for training the WM--MS3M PRB-conditioned world model.

Usage:
    python scripts/train_wm_ms3m.py --data-dir ./data --ckpt-dir ./artifacts
"""

import argparse
from wm_ms3m import WMMS3MConfig, train_and_eval_wm_ms3m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing x.npy and y.npy")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory to store checkpoints & metrics")
    args = parser.parse_args()

    cfg = WMMS3MConfig()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir

    _ = train_and_eval_wm_ms3m(cfg)


if __name__ == "__main__":
    main()
