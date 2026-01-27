#!/usr/bin/env python3
"""
Simple CLI wrapper for verl's native trainer.
This script calls verl.trainer.main_ppo with GRPO configuration.

Usage:
    python train_verl.py --config configs/verl_config.yaml
    
Or directly use verl's CLI:
    python -m verl.trainer.main_ppo --config configs/verl_config.yaml
"""

import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-8B to generate harder questions using verl GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_verl.py --config configs/verl_config.yaml
  
  # With additional verl arguments
  python train_verl.py --config configs/verl_config.yaml --training.train_batch_size 8
  
  # Direct verl CLI (equivalent)
  python -m verl.trainer.main_ppo --config configs/verl_config.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/verl_config.yaml",
        help="Path to verl configuration file",
    )
    parser.add_argument(
        "--seed_questions",
        type=str,
        default="data/seed_questions.txt",
        help="Path to seed questions file (for reference - actual data should be in verl config)",
    )
    
    args, unknown_args = parser.parse_known_args()
    
    # Check if verl is available
    try:
        import verl
    except ImportError:
        print("ERROR: verl is not installed. Please install it:")
        print("  pip install verl[vllm,sglang]")
        print("  or from source: git clone https://github.com/volcengine/verl.git && cd verl && pip install -e .")
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print(f"Please create a verl configuration file or specify an existing one with --config")
        sys.exit(1)
    
    # Prepare verl command
    verl_cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config", args.config,
    ]
    
    # Add any additional verl arguments (these will be passed to verl)
    verl_cmd.extend(unknown_args)
    
    print("=" * 70)
    print("Starting verl GRPO training")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Seed questions reference: {args.seed_questions}")
    print(f"Command: {' '.join(verl_cmd)}")
    if unknown_args:
        print(f"Additional args: {' '.join(unknown_args)}")
    print("=" * 70)
    print()
    print("Note: This script is a simple wrapper around verl's native trainer.")
    print("You can also run verl directly:")
    print(f"  python -m verl.trainer.main_ppo --config {args.config}")
    print()
    
    # Run verl trainer
    try:
        subprocess.run(verl_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: verl training failed with exit code {e.returncode}")
        print("Check verl logs above for details.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
