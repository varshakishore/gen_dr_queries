# Quick Start Guide

## Simplest Way to Train

Use verl's native CLI directly:

```bash
python -m verl.trainer.main_ppo --config configs/verl_config.yaml
```

That's it! verl handles everything.

## What You Need

1. **Install verl**:
   ```bash
   pip install verl[vllm,sglang]
   ```

2. **Set OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Configure verl**:
   Edit `configs/verl_config.yaml` to set:
   - Model path
   - Group size (number of rollouts per seed)
   - Training hyperparameters
   - Reward model settings

4. **Prepare data**:
   - Put seed questions in the format verl expects (check verl docs for dataset format)
   - Or configure verl to load from your `data/seed_questions.txt`

## Using the Wrapper Script

If you prefer a simple wrapper:

```bash
python train_verl.py --config configs/verl_config.yaml
```

This just calls verl's CLI with your config.

## Custom Reward Model

The project includes a custom reward model (`reward_model.py`) that uses GPT-5 judge. To use it with verl:

1. Make sure verl can import it (add project root to PYTHONPATH if needed)
2. Reference it in your verl config:
   ```yaml
   reward_model:
     type: "custom"
     module: "reward_model"
     class: "VerlRewardModel"
   ```

Or configure verl to use it via its reward model interface.

## Why This Approach?

- **Less code to maintain**: verl handles all the training logic
- **Always up-to-date**: verl improvements automatically benefit you
- **Standard**: Follows verl's recommended usage
- **Flexible**: All verl features available via config

## That's It!

This project uses verl's native CLI exclusively. All training is done through verl's standard interface, keeping the codebase simple and maintainable.
