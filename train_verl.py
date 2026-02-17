#!/usr/bin/env python3
"""
Simple CLI wrapper for verl's native trainer.
Uses verl's default config (like the official examples) and applies our YAML as
Hydra overrides, so we don't need to duplicate ray_kwargs, transfer_queue, etc.

Usage:
    source secrets.env   # export OPENAI_API_KEY etc.
    python train_verl.py --config configs/verl_config.yaml
    python train_verl.py --config configs/verl_config.yaml data.train_batch_size=8
"""

import sys
import os
import subprocess
import argparse
import yaml

# Add project root to Python path so verl can import reward_model
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _flatten_to_overrides(obj, prefix=""):
    """Convert nested dict to Hydra override strings key=value."""
    overrides = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            overrides.extend(_flatten_to_overrides(v, key))
    elif isinstance(obj, list):
        overrides.append(f"{prefix}=[{','.join(str(x) for x in obj)}]")
    elif obj is None:
        overrides.append(f"{prefix}=null")
    else:
        val = obj
        if isinstance(val, str) and (" " in val or "=" in val or val == ""):
            val = f'"{val}"'
        elif isinstance(val, bool):
            val = "true" if val else "false"
        overrides.append(f"{prefix}={val}")
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Train with verl GRPO using default config + our overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_verl.py --config configs/verl_config.yaml
  python train_verl.py --config configs/verl_config.yaml data.train_batch_size=8
  python train_verl.py  # uses configs/verl_config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/verl_config.yaml",
        help="YAML with overrides (verl default config is used as base)",
    )
    parser.add_argument(
        "--seed_questions",
        type=str,
        default="data/seed_questions.txt",
        help="Path to seed questions file (for reference)",
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
    
    config_file = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.exists(config_file):
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    with open(config_file) as f:
        config = yaml.safe_load(f)
    if not config:
        config = {}
    overrides = _flatten_to_overrides(config)

    # No --config-path: use verl's default config (like official examples)
    verl_cmd = [sys.executable, "-m", "verl.trainer.main_ppo"]
    verl_cmd.extend(overrides)
    verl_cmd.extend(unknown_args)

    print("=" * 70)
    print("Starting verl GRPO training (verl default config + overrides)")
    print("=" * 70)
    print(f"Override config: {config_file}")
    print(f"Seed questions reference: {args.seed_questions}")
    print(f"Command: {' '.join(verl_cmd[:4])} ... ({len(overrides)} overrides + {len(unknown_args)} extra)")
    if unknown_args:
        print(f"Extra args: {' '.join(unknown_args)}")
    print("=" * 70)
    print()

    # Run verl trainer from project root so data paths and reward_model resolve correctly
    try:
        subprocess.run(verl_cmd, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: verl training failed with exit code {e.returncode}")
        print("Check verl logs above for details.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
