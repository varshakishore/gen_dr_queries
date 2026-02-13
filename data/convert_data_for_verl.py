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

"""
Your task is to generate hard deep research questions that require searching for information in mutiple sources, reasoning and synthesizing the information, to provide a detailed answer.
Given a seed research question, generate a strictly harder research question that:
- Requires deeper reasoning, synthesis, or abstraction
- Cannot be answered by a surface-level or textbook response
- Remains within the same general topic or problem space as the seed question
- May reframe the problem, but must not simplify it


Here are some strategies to make the question harder:
- Adding additional constrains and requiring specifics
- altering the question to make it unanswerable
- changing the question so that the retrieved evidence might have conflicts
- changing the question such that the retrieved evidence might be low quality
- constructing a question such that it requires multi-hop reasoning or synsthesis from multiple sources
These strategoes are not exhaustive, and you can come up with other strategies to make the question harder.
- Questions that have a false premise

Return a JSON object with exactly the following fields:
- "rationale": A brief explanation (1–3 sentences) of why the new question is strictly harder than the seed question
- "harder_question": The new, strictly harder research question

Do not include any text outside the JSON object.

Seed question:
{SEED_QUESTION}
"""

f"""
Deep research queries require searching for information from multiple sources, reasoning about the retrieved results and synthesizing a structured report.
Given a seed question, I want to generate a strictly harder deep research question.

The generated question must be significantly more demanding than the seed question in terms of reasoning, evidence integration, and epistemic difficulty.

Given a seed research question, generate ONE new research question that:
- Remains within the same general topic or problem space as the seed question
- Is strictly more difficult in scope, reasoning, or evidence requirements
- Cannot be adequately answered with a definitional, summary, or single-source response

Difficulty may be increased using strategies such as:
- Adding precise constraints (e.g., time periods, populations, mechanisms, evaluation criteria)
- Requiring reconciliation of conflicting, low-quality, or incomplete evidence
- By adding a false premise to the question
- Requiring multi-hop reasoning or synthesis across documents
- Altering the question to make it unanswerable

These strategies are illustrative, not exhaustive.

The output MUST be a JSON object with exactly the following fields:
- "rationale": 1–2 sentences explaining why the new question is strictly harder
- "harder_question": The newly generated, strictly harder research question

The new question should sound natural, be concise, and remain broadly framed.

No additional text outside the JSON object is allowed.

Example: 
Seed question:
is socail media good or bad for us?
Output:
{{
    "rationale": "Adding specific contraints and requiring the model to make a judgement based on the retrieved evidence",
    "harder_question": "Discuss the societal impact of social media. Look at some of the pros and cons it has in different areas, like mental health and politics. Based on this, has its effect been mostly positive or mostly negative?"
}}
"""



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
    
    for i, question in enumerate(questions):
        # Create prompt in HuggingFace chat template format with instruction
        prompt = prompt_template(question)
        
        # Create verl data entry (extra_info must be a dict; VeRL expects .get("extra_info", {}).get("index", 0))
        entry = {
            "data_source": data_source,
            "prompt": prompt,  # This will be serialized as JSON string in Parquet
            "ability": ability,
            "reward_model": {
                "ground_truth": question  # Store original question as ground truth
            },
            "extra_info": {"index": i},
        }
        
        verl_data.append(entry)
    
    return verl_data

def save_parquet(data, output_file):
    """Save data to Parquet format.
    VeRL expects extra_info and reward_model as dicts (e.g. .get('reward_model', {}).get('ground_truth')),
    so we store them as dict columns (not JSON strings); pandas/pyarrow writes them as structs.
    """
    df = pd.DataFrame([
        {
            "data_source": e["data_source"],
            "prompt": json.dumps(e["prompt"]),
            "ability": e["ability"],
            "reward_model": e["reward_model"],  # dict → struct, so VeRL gets .get("ground_truth")
            "extra_info": e["extra_info"],  # dict → struct, so VeRL gets .get("index", 0)
        }
        for e in data
    ])
    df.to_parquet(output_file, index=False, engine="pyarrow")
    print(f"✓ Saved {len(data)} entries to {output_file}")

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