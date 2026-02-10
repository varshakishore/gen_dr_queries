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

# Add project root to Python path so verl can import reward_model
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-8B to generate harder questions using verl GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_verl.py --config configs/verl_config.yaml
  
  # With Hydra overrides (use key=value, not --key value)
  python train_verl.py --config configs/verl_config.yaml training.train_batch_size=8
  
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
    
    # Hydra expects config-path (directory) and config-name (filename without .yaml)
    config_path = os.path.dirname(args.config) or "."
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    # Prepare verl command (verl uses Hydra: --config-path, --config-name)
    verl_cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-path", config_path,
        "--config-name", config_name,
    ]
    
    # Add any additional Hydra overrides (e.g. training.train_batch_size=8)
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
    print("You can also run verl directly (Hydra uses --config-path and --config-name):")
    print(f"  python -m verl.trainer.main_ppo --config-path {config_path} --config-name {config_name}")
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
