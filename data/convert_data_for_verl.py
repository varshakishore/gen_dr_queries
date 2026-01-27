#!/usr/bin/env python3
"""
Convert seed questions to verl's expected data format.

verl expects data in Parquet format with:
- prompt: HuggingFace chat template format [{"role": "user", "content": "question"}]
- data_source: dataset identifier
- ability: task category
- reward_model: dict with ground_truth field
- extra_info: optional metadata

Usage:
    python convert_data_for_verl.py --input data/seed_questions.txt --output data/seed_questions.parquet
"""

import argparse
import json
from pathlib import Path
import pandas as pd

def load_seed_questions(input_file):
    """Load seed questions from text file (one per line)."""
    questions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                questions.append(line)
    return questions

def convert_to_verl_format(questions, data_source="gen_dr_queries", ability="question_generation", prompt_template=None):
    """
    Convert questions to verl's expected format.
    
    Args:
        questions: List of question strings
        data_source: Dataset identifier
        ability: Task category
        prompt_template: Optional function that takes a question and returns a prompt list.
                         If None, uses default template with instruction to generate harder question.
    
    Returns:
        List of dicts in verl format
    """
    verl_data = []
    
    # Default prompt template: instruct model to generate a harder version of the question
    if prompt_template is None:
        def default_prompt_template(question):
            return [
                {"role": "system", "content": "You are an expert at creating challenging questions. Given a seed question, generate a harder version that requires deeper understanding or more complex reasoning."},
                {"role": "user", "content": f"Generate a harder version of the following question:\n\n{question}"}
            ]
        prompt_template = default_prompt_template
    
    for question in questions:
        # Create prompt in HuggingFace chat template format with instruction
        prompt = prompt_template(question)
        
        # Create verl data entry
        entry = {
            "data_source": data_source,
            "prompt": prompt,  # This will be serialized as JSON string in Parquet
            "ability": ability,
            "reward_model": {
                "ground_truth": question  # Store original question as ground truth
            },
            "extra_info": {}
        }
        
        verl_data.append(entry)
    
    return verl_data

def save_parquet(data, output_file):
    """Save data to Parquet format."""
    # Convert prompt (list of dicts) to JSON string for Parquet compatibility
    df_data = []
    for entry in data:
        df_entry = {
            "data_source": entry["data_source"],
            "prompt": json.dumps(entry["prompt"]),  # Serialize as JSON string
            "ability": entry["ability"],
            "reward_model": json.dumps(entry["reward_model"]),  # Serialize as JSON string
            "extra_info": json.dumps(entry["extra_info"])  # Serialize as JSON string
        }
        df_data.append(df_entry)
    
    df = pd.DataFrame(df_data)
    df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"✓ Saved {len(df_data)} entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert seed questions to verl's expected Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python convert_data_for_verl.py --input data/seed_questions.txt --output data/seed_questions.parquet
  
  # With custom data source
  python convert_data_for_verl.py --input data/seed_questions.txt --output data/seed_questions.parquet --data_source my_dataset
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/seed_questions.txt",
        help="Input text file with seed questions (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/seed_questions.parquet",
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="gen_dr_queries",
        help="Dataset identifier for verl"
    )
    parser.add_argument(
        "--ability",
        type=str,
        default="question_generation",
        help="Task category/ability name"
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    # Load questions
    print(f"Loading questions from {args.input}...")
    questions = load_seed_questions(args.input)
    print(f"✓ Loaded {len(questions)} questions")
    
    # Convert to verl format
    print("Converting to verl format...")
    verl_data = convert_to_verl_format(
        questions,
        data_source=args.data_source,
        ability=args.ability
    )
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet
    print(f"Saving to {args.output}...")
    save_parquet(verl_data, args.output)
    
    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)
    print(f"Input:  {args.input} ({len(questions)} questions)")
    print(f"Output: {args.output}")
    print(f"\nNext steps:")
    print(f"1. Update verl_config.yaml to point to: {args.output}")
    print(f"2. Run: python train_verl.py --config configs/verl_config.yaml")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())