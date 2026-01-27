# Quick Reference Guide

## Installation

```bash
# 1. Install verl
pip install verl[vllm,sglang]

# 2. Install project dependencies
pip install -r requirements.txt

# 3. Set OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

## Running Training

### Simplest Way

```bash
python train_verl.py --config configs/verl_config.yaml
```

### Direct verl CLI (Equivalent)

```bash
python -m verl.trainer.main_ppo --config configs/verl_config.yaml
```

## Key Files

- **`train_verl.py`**: Main entry point (wrapper around verl CLI)
- **`configs/verl_config.yaml`**: verl configuration file
- **`reward_model.py`**: Custom reward model using GPT-5 judge
- **`data/seed_questions.txt`**: Seed questions (may need format conversion)

## Configuration

Edit `configs/verl_config.yaml` to set:
- Model path
- Group size (rollouts per seed)
- Batch size, learning rate
- Backend (FSDP/Megatron)
- Rollout backend (vLLM/SGLang)

## Important Notes

1. **Data Format**: verl expects data in a specific format - you may need to convert `seed_questions.txt`
2. **Reward Model**: Configure verl to use `reward_model.py` (check verl docs for exact format)
3. **verl Config**: Follow verl's configuration format exactly
4. **Check verl Docs**: For dataset format, reward model setup, and advanced options

## Troubleshooting

- **verl not found**: `pip install verl[vllm,sglang]`
- **Import errors**: Add project to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
- **API errors**: Check `OPENAI_API_KEY` is set correctly

## Full Documentation

See `USAGE_INSTRUCTIONS.md` for detailed instructions.
