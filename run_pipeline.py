#!/usr/bin/env python
"""
run_pipeline.py — Master orchestrator script for the FIFA 26 Predictor.
Runs the entire pipeline sequentially.

Usage:
  python run_pipeline.py                   # Run the full pipeline (including scraping)
  python run_pipeline.py --skip-scraping   # Skip the web scraping phase and run processing + training + predictions
  python run_pipeline.py --only-train      # Run only training and evaluation
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent

def run_step(script_path: str, args: list = None) -> bool:
    """Run a Python script as a subprocess and monitor execution."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    print(f"\n======================================================================")
    print(f"[*] Running: {' '.join(cmd)}")
    print(f"======================================================================")
    
    start_time = time.time()
    try:
        # Run process and stream output to console
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
                
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"[OK] Step completed successfully in {elapsed:.1f}s.")
            return True
        else:
            print(f"[FAIL] Step failed with exit code {process.returncode} after {elapsed:.1f}s.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="FIFA 26 Predictor Pipeline Orchestrator")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip web scraping steps and start at data merge")
    parser.add_argument("--only-train", action="store_true", help="Skip scraping and processing, only run train + evaluate")
    args = parser.parse_args()

    pipeline = []

    # Define steps
    scrape_lineups_step = ("src/scraping/scrape_lineups.py", [])
    scrape_stats_step = ("src/scraping/scrape_player_stats.py", [])
    merge_step = ("src/processing/merge_data.py", [])
    feat_step = ("src/processing/feature_engineering.py", [])
    train_step = ("src/train.py", ["--model", "all"])
    train_xgb_step = ("src/train_xgb.py", [])
    eval_step = ("src/evaluate.py", [])
    train_ensemble_step = ("src/train_ensemble.py", [])  # learn blend weights after eval
    predict_step = ("src/predict_2026.py", [])
    ensemble_step = ("src/ensemble_2026.py", [])

    if args.only_train:
        print("[START] Starting pipeline (TRAINING ONLY mode)...")
        pipeline = [train_step, train_xgb_step, eval_step, train_ensemble_step]
    elif args.skip_scraping:
        print("[START] Starting pipeline (SKIPPING SCRAPING mode)...")
        pipeline = [merge_step, feat_step, train_step, train_xgb_step, eval_step, train_ensemble_step, predict_step, ensemble_step]
    else:
        print("[START] Starting FULL pipeline (including web scraping)...")
        pipeline = [
            scrape_lineups_step,
            scrape_stats_step,
            merge_step,
            feat_step,
            train_step,
            train_xgb_step,
            eval_step,
            train_ensemble_step,
            predict_step,
            ensemble_step
        ]

    total_steps = len(pipeline)
    for idx, (script, script_args) in enumerate(pipeline, 1):
        print(f"\n[Step {idx}/{total_steps}] Processing {script}...")
        success = run_step(script, script_args)
        if not success:
            print(f"\n[ERROR] Pipeline stopped due to failure at step: {script}")
            sys.exit(1)

    print("\n======================================================================")
    print("[SUCCESS] All pipeline steps executed successfully!")
    print("======================================================================")

if __name__ == "__main__":
    main()
