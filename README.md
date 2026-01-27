# gen_dr_queries

This project trains a Qwen3-8B model to generate harder questions from seed questions using Group Relative Policy Optimization (GRPO) with the **verl** RL framework. GPT-5 (or GPT-4) is used as a judge to evaluate the quality and difficulty of generated questions.

**Recommended Approach**: Use verl's native CLI (`python -m verl.trainer.main_ppo`) directly with a configuration file. This is simpler, more maintainable, and leverages verl's full capabilities.

## Overview

The system implements a reinforcement learning pipeline where:

1. **Seed Questions** are loaded from a text file
2. **Qwen3-8B Model** generates multiple candidate harder questions for each seed
3. **GPT-5 Judge** scores each candidate on difficulty, clarity, and relevance
4. **GRPO Algorithm** updates the model using group-relative advantages
5. **Training Loop** orchestrates the process using verl framework

## Architecture

```
Seed Question → Qwen3-8B → Generate G Candidates → GPT-5 Judge → Compute Rewards → GRPO Update → Updated Model
```

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA ≥ 12.8 (for latest verl versions)
- PyTorch ≥ 2.0.0

### Install verl (Required)

verl is required for this project. Install it using one of the following methods:

**Option 1: Install from pip**
```bash
pip install verl
```

**Option 2: Install with backends (recommended)**
```bash
# With vLLM backend (recommended for fast rollouts)
pip install verl[vllm]

# With SGLang backend
pip install verl[sglang]

# With both backends
pip install verl[vllm,sglang]
```

**Option 3: Install from source**
```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .[vllm,sglang]
```

### Install Project Dependencies

```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key for the judge:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. (Optional) If you have access to Qwen3-8B or GPT-5, update the model names in `configs/verl_config.yaml`

## Usage

### Recommended: Use verl's Native CLI

The simplest way to train is using verl's native CLI directly:

```bash
python -m verl.trainer.main_ppo --config configs/verl_config.yaml
```

Or use the provided wrapper script:

```bash
python train_verl.py --config configs/verl_config.yaml
```


### With Custom verl Configuration

```bash
python -m verl.trainer.main_ppo --config path/to/your/verl_config.yaml
```

### With Additional verl Arguments

You can pass any verl CLI arguments:

```bash
python -m verl.trainer.main_ppo \
    --config configs/verl_config.yaml \
    --training.train_batch_size 8 \
    --training.learning_rate 2e-5
```

### Resume Training

verl handles checkpointing automatically. Check verl documentation for resume options.

### Why Use verl's Native CLI?

Using `python -m verl.trainer.main_ppo` directly provides:
- **Simplicity**: No custom wrapper code to maintain
- **Full verl features**: Access to all verl capabilities via config
- **Better maintenance**: Updates to verl automatically benefit your project
- **Standard approach**: Follows verl's recommended usage pattern
- **Flexibility**: All verl CLI options and config overrides available

### Custom Configuration

Edit `configs/default_config.yaml` or create your own config file:

```yaml
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  
generation:
  num_candidates: 4  # Group size G for GRPO
  temperature: 1.0
  
grpo:
  learning_rate: 1e-5
  
training:
  num_epochs: 10
  batch_size: 4
```


## Configuration

Key configuration parameters:

- **Group Size (G)**: Number of candidates generated per seed (typically 4-16)
- **Learning Rate**: Policy update step size (typically 1e-6 to 1e-5)
- **Temperature**: Sampling temperature for generation (0.7-1.2)
- **Max Tokens**: Maximum length of generated questions

## Project Structure

```
gen_dr_queries/
├── data/
│   └── seed_questions.txt          # Seed questions (one per line)
├── models/
│   └── judge.py                    # GPT-5 judge API wrapper
├── rl/
│   └── reward.py                   # Reward computation utilities
├── reward_model.py                 # Custom reward model for verl (GPT-5 judge)
├── train_verl.py                   # CLI wrapper for verl's native trainer
├── configs/
│   └── verl_config.yaml            # verl configuration file
├── requirements.txt
├── README.md                        # This file
├── QUICKSTART.md                    # Quick start guide
└── VERL_USAGE.md                    # Detailed verl usage guide
```

## GRPO Algorithm

Group Relative Policy Optimization (GRPO) is a policy gradient method that:

1. Generates multiple candidates (group) for each seed question
2. Scores each candidate with the judge
3. Normalizes rewards within the group: `advantage = (reward - mean) / std`
4. Updates policy: `loss = -mean(advantage * log_prob)`

This removes the need for a separate baseline network.

This project uses **verl's built-in GRPO implementation**, which provides optimized performance, distributed training support, and production-ready code.

See [VERL_USAGE.md](VERL_USAGE.md) for detailed verl usage information.

## Judge Evaluation

The GPT-5 judge evaluates each candidate on:

- **Difficulty** (50% weight): How much harder than the seed
- **Clarity** (25% weight): Well-formed and understandable
- **Relevance** (25% weight): Stays on the same topic

## Output

Training produces:

- **Checkpoints**: Saved in `checkpoints/` directory
- **Logs**: Training logs in `logs/` directory
- **Metrics**: Loss, rewards, advantages tracked during training

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- OpenAI API access for judge

## Notes

- The model uses Qwen2.5-7B-Instruct as Qwen3-8B may not be publicly available
- The judge uses GPT-4o as GPT-5 may not be available yet
- Adjust model names in config if you have access to different models
- Monitor API costs when using GPT-5 judge for many candidates

## License

[Add your license here]
