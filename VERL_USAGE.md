# Using verl for GRPO Training

This guide explains how to use verl framework for GRPO training in this project.

**Note**: verl is **required** for this project. This project uses verl's GRPO implementation exclusively.

## Overview

verl (Volcano Engine Reinforcement Learning) is a comprehensive RL framework for LLMs that provides:
- Built-in GRPO algorithm implementation
- Efficient rollout management (vLLM, SGLang backends)
- Distributed training support (FSDP, Megatron)
- Optimized memory management and batching

## Installation

### Basic Installation

```bash
pip install verl
```

### Installation with Backends

For optimal performance, install verl with rollout backends:

```bash
# With vLLM backend (recommended for fast rollouts)
pip install verl[vllm]

# With SGLang backend
pip install verl[sglang]

# With both
pip install verl[vllm,sglang]
```

### From Source

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .[vllm,sglang]
```

### Requirements

- Python ≥ 3.10
- CUDA ≥ 12.8 (for latest versions)
- PyTorch ≥ 2.0.0
- Sufficient GPU memory for your model

## Usage

### Basic Command

```bash
python train.py \
    --config configs/default_config.yaml \
    --seed_questions data/seed_questions.txt \
    --use_verl \
    --verl_config configs/verl_config.yaml
```

### Configuration

The `configs/verl_config.yaml` file contains verl-specific settings:

```yaml
algorithm:
  type: "grpo"
  adv_estimator: "grpo"

actor_rollout_ref:
  rollout:
    n: 4  # Group size - must be > 1 for GRPO
  actor:
    loss_agg_mode: "token-mean"  # or "sample-mean"

training:
  backend: "fsdp"  # Options: fsdp, megatron
  rollout_backend: "vllm"  # Options: vllm, sglang
```

### Key verl Features Used

1. **Group Rollouts**: verl automatically handles generating multiple candidates per seed
2. **Reward Model Integration**: Our `VerlRewardModel` wrapper makes GPT-5 judge compatible with verl
3. **GRPO Algorithm**: verl's built-in GRPO handles advantage computation and policy updates
4. **Efficient Batching**: verl optimizes batch processing for large-scale training

## verl API Compatibility

verl's API may vary by version. The `VerlGRPOTrainer` class includes flexible initialization that tries multiple API patterns:

1. Direct config-based initialization
2. Separate argument initialization
3. Minimal initialization

If you encounter API errors, check:
- verl version: `pip show verl`
- verl documentation: https://github.com/volcengine/verl
- verl examples: Check verl's `examples/` directory

## verl is Required

verl is a required dependency for this project. The training will fail if verl is not properly installed. Make sure to install verl before running training:

```bash
pip install verl[vllm,sglang]
```

## Troubleshooting

### Import Errors

If you see `ImportError` for verl:
```bash
pip install verl
# or
pip install verl[vllm,sglang]
```

### API Errors

If verl trainer initialization fails:
1. Check verl version compatibility
2. Review verl's latest documentation
3. Check `configs/verl_config.yaml` format
4. verl is required - training will fail if verl is not properly installed

### Backend Errors

If you see errors about vLLM or SGLang:
- Install the specific backend: `pip install verl[vllm]`
- Or use a different backend in config (e.g., change `rollout_backend`)

## Benefits of Using verl

1. **Performance**: Optimized rollout generation and batching
2. **Scalability**: Built-in support for distributed training
3. **Maintenance**: Well-maintained, production-ready code
4. **Features**: Access to latest RL algorithms and optimizations
5. **Community**: Active development and support

## Why verl?

This project uses verl's GRPO implementation because it provides:
- **Better Performance**: Optimized rollout generation and batching
- **Scalability**: Built-in support for distributed training
- **Maintenance**: Well-maintained, production-ready code
- **Features**: Access to latest RL algorithms and optimizations
- **Community**: Active development and support

verl's GRPO implementation is battle-tested and optimized for large-scale LLM training.
