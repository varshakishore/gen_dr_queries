# Usage Instructions

## Overview

This project trains a Qwen3-8B model to generate harder questions using GRPO (Group Relative Policy Optimization) via verl framework. The project uses verl's native CLI interface, keeping the codebase minimal and maintainable.

## Prerequisites

1. **Python ≥ 3.10**
2. **CUDA ≥ 12.8** (for latest verl versions)
3. **PyTorch ≥ 2.0.0**
4. **OpenAI API key** (for GPT-5/GPT-4 judge)

## Step 1: Installation

### Install verl (Required)

```bash
# Option 1: Basic installation
pip install verl

# Option 2: With backends (recommended for better performance)
pip install verl[vllm,sglang]

# Option 3: From source
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .[vllm,sglang]
```

### Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Step 2: Configure verl

Edit `configs/verl_config.yaml` to configure:

### Key Settings:

1. **Model**: Set the model path
   ```yaml
   model:
     name: "Qwen/Qwen2.5-7B-Instruct"  # Or your model path
     torch_dtype: "float16"
   ```

2. **GRPO Algorithm**: Already configured
   ```yaml
   algorithm:
     type: "grpo"
     adv_estimator: "grpo"
   ```

3. **Group Size**: Number of rollouts per seed (must be > 1 for GRPO)
   ```yaml
   actor_rollout_ref:
     rollout:
       n: 4  # Increase for more stable training
   ```

4. **Training Parameters**:
   ```yaml
   training:
     train_batch_size: 4
     learning_rate: 1e-5
     backend: "fsdp"  # or "megatron"
     rollout_backend: "vllm"  # or "sglang"
   ```

5. **Data**: Configure your dataset path in verl config
   - verl expects data in a specific format (check verl documentation)
   - You may need to convert `data/seed_questions.txt` to verl's expected format

### Configure Custom Reward Model

To use the GPT-5 judge reward model, you need to configure verl to use it. The exact configuration depends on verl's version and API. Generally, you'll need to:

1. Make sure verl can import `reward_model.py`:
   - Add project root to PYTHONPATH, or
   - Install the project as a package

2. Reference it in verl config (check verl docs for exact format):
   ```yaml
   reward_model:
     type: "custom"
     module: "reward_model"
     class: "VerlRewardModel"
     # Or use verl's reward model configuration format
   ```

**Note**: verl's reward model configuration format may vary. Check verl's documentation for the exact way to specify custom reward models.

## Step 3: Prepare Data

verl expects data in a specific format. You have two options:

### Option A: Convert to verl's format

Check verl documentation for the expected dataset format and convert `data/seed_questions.txt` accordingly.

### Option B: Use verl's dataset utilities

Use verl's built-in dataset loading utilities to load from your text file.

## Step 4: Run Training

### Method 1: Using the Wrapper Script (Recommended)

```bash
python train_verl.py --config configs/verl_config.yaml
```

### Method 2: Direct verl CLI

```bash
python -m verl.trainer.main_ppo --config configs/verl_config.yaml
```

### Method 3: With Additional verl Arguments

You can pass any verl CLI arguments:

```bash
python train_verl.py \
    --config configs/verl_config.yaml \
    --training.train_batch_size 8 \
    --training.learning_rate 2e-5
```

Or directly:

```bash
python -m verl.trainer.main_ppo \
    --config configs/verl_config.yaml \
    --training.train_batch_size 8
```

## Step 5: Monitor Training

verl handles logging automatically. Check:
- Console output for training progress
- verl's default log directory (check verl docs)
- WandB if configured in verl config

## Important Notes

### 1. Reward Model Integration

The custom reward model (`reward_model.py`) uses GPT-5 judge. To integrate it with verl:

- **Check verl's documentation** for how to specify custom reward models
- The reward model interface expects `(prompts, responses) -> rewards`
- Make sure `OPENAI_API_KEY` is set

### 2. Data Format

- verl has specific data format requirements
- You may need to convert `data/seed_questions.txt` to verl's expected format
- Check verl's dataset documentation

### 3. Configuration

- All training parameters are in `configs/verl_config.yaml`
- You can override any parameter via CLI arguments
- Follow verl's configuration format exactly

### 4. Model Loading

- verl handles model loading automatically
- Make sure the model path in config is correct
- verl supports HuggingFace models directly

### 5. Checkpointing

- verl handles checkpointing automatically
- Check verl documentation for checkpoint paths and resume options

## Troubleshooting

### verl Not Found

```bash
pip install verl[vllm,sglang]
```

### Import Errors

Make sure the project root is in PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Reward Model Not Found

- Check that `reward_model.py` is importable
- Verify verl config references it correctly
- Check verl's documentation for custom reward model setup

### Data Format Issues

- Check verl's dataset format requirements
- Convert your data to verl's expected format
- Use verl's dataset utilities if available

### API Errors

- Verify `OPENAI_API_KEY` is set
- Check API key is valid
- Monitor API rate limits

## Project Structure

```
gen_dr_queries/
├── train_verl.py              # CLI wrapper (use this)
├── reward_model.py            # Custom reward model (GPT-5 judge)
├── models/
│   └── judge.py              # GPT-5 judge implementation
├── rl/
│   └── reward.py             # Reward computation utilities
├── configs/
│   └── verl_config.yaml     # verl configuration
├── data/
│   └── seed_questions.txt   # Seed questions (may need conversion)
└── requirements.txt         # Dependencies
```

## Next Steps

1. **Read verl documentation**: Understand verl's configuration format and requirements
2. **Configure reward model**: Set up verl to use `reward_model.py`
3. **Prepare data**: Convert data to verl's format
4. **Start training**: Run with `python train_verl.py --config configs/verl_config.yaml`
5. **Monitor**: Watch training progress and adjust hyperparameters as needed

## Additional Resources

- verl GitHub: https://github.com/volcengine/verl
- verl Documentation: Check verl's docs for latest API
- GRPO Paper: For understanding the algorithm
