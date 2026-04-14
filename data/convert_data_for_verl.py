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
from datasets import load_dataset

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

"""Deep research queries require searching for information from multiple sources, reasoning about the retrieved results and synthesizing a structured report.
Given a seed question, I want to generate a strictly harder deep research question.

The generated question must be significantly more demanding than the seed question in terms of reasoning, evidence integration, and epistemic difficulty.

Given a seed research question, generate ONE new research question that:
- Remains within the same general topic or problem space as the seed question
- Is strictly more difficult in scope, reasoning, or evidence requirements
- Cannot be adequately answered with a definitional, summary, or single-source response

Difficulty may be increased using strategies such as:
- Adding precise constraints (e.g., time periods, populations, mechanisms, evaluation criteria)
- Requiring reconciliation of conflicting, low-quality, or incomplete evidence
- Introducing a subtle or false premise
- Requiring multi-hop reasoning or synthesis across documents
- Altering the question to make it partially or fully unanswerable
- Requiring analysis of experimental design details rather than surface results
- Narrowing to rare or sparsely studied subcases within the same topic

These strategies are illustrative, not exhaustive.

The output MUST be a JSON object with exactly the following fields:
- "rationale": 1–2 sentences explaining why the new question is strictly harder
- "harder_question": The newly generated, strictly harder research question

The new question should sound natural, be concise, and remain broadly framed.

No additional text outside the JSON object is allowed.

Example 1:
Seed question:
is social media good or bad for us?
Output:
{{
    "rationale": "Adding specific constraints and requiring a judgment based on synthesizing evidence across domains makes the question harder.",
    "harder_question": "Across mental health, political polarization, and economic opportunity, has social media had a net positive or net negative societal impact since 2010, and how should conflicting empirical findings be reconciled?"
}}

Example 2:
Seed question:
Which DPO variants have been evaluated on language models with fewer than 1B parameters?
Output:
{{
    "rationale": "I am going to require analyzing experimental setups, dataset characteristics, and evaluation setups rather than simply listing variants.",
    "harder_question": "Given 3,000 pairwise judgments on long-form question answering tasks (10,000–50,000 tokens per instance) and a sub-1B parameter model, which DPO variant is most suitable, considering reported accuracy on comparable tasks and the rigor of its evaluation methodology?"
}}

Example 3:
Seed question:
Tell me about rubric generation models.
Output:
{{
    "rationale": "I am going to narrow the scope to models explicitly trained for rubric generation, requiring identification of specific training setups rather than general usage.",
    "harder_question": "Tell me about models trained specifically for rubric generation?"
}}

Example 4:
Seed question:
what body regions encode immune memory
Output:
{{
    "rationale": "The harder question will include a biologically implausible premise, requiring the system to detect and reason about the false assumption.",
    "harder_question": "What brain regions encode immune memory?"
}}

Example 5:
Seed question:
What proteins are upregulated in early-stage Alzheimers but downregulated in late-stage disease?
Output:
{{
    "rationale": "The harder question removes the explicit contrast and instead requires identifying consistencies across disease stages, which is less directly documented and requires broader synthesis.",
    "harder_question": "What proteins are upregulated in early-stage Alzheimers and late-stage Alzheimers?"
}}
"""

current_prompt = """
You are an expert in constructing challenging research questions.

Given a seed question, produce an updated question designed to expose weaknesses in deep research systems.

RULES:
- Avoid "what" questions that merely retrieve information.
- The updated question MUST require higher-order thinking: analysis, comparison, evaluation, or synthesis.
- It must NOT be answerable with a definition, summary, or single-source response.
- The updated question length should change by fewer than 10 words from the seed.

STRATEGIES TO CONSIDER (pick the most promising one):
1. Require synthesis across 5+ sources or clearly disjoint domains (e.g., science + economics).
2. Require synthesis across differing viewpoints, stakeholder incentives, or theoretical frameworks.
3. Require multi-step reasoning, structured argumentation, or hierarchical planning.
4. Require handling conflicting, incomplete, or low-quality evidence.
5. Require correcting a hidden misconception or establishing key knowns before answering.
6. Ask for a specific "moment of truth" — a concrete case highlighting consequences and lessons learned.
7. Embed a specific context that changes the answer (e.g., "explain to a policymaker with no ML background").

OUTPUT FORMAT (valid JSON, no extra text):
{{
  "brainstorming": "<think about distinct strategies and reason about why they may or may not work>",
  "chosen_strategy": "<name and justify the single most promising strategy>",
  "updated_question": "<the rewritten question>",
  "why_harder": "<explanation of why this question might be hard for a deep research system>",
  "verification_criterion": "<one concrete, testable criterion for checking whether the answer is good>"
}}

EXAMPLES:

Seed question: What body regions encode immune memory?
Output:
{{
  "brainstorming": "My first instinct is to push for cross-domain synthesis — force the question to span immunology and neuroscience simultaneously. But that might actually make it easier, since a deep research system could just pull from neuroimmunology literature and produce a fluent-sounding answer without real reasoning. What would be harder is embedding a false premise directly into the question — attributing immune memory to the brain — so the system has to detect the error before it can even begin answering. A shallow system will likely just hallucinate brain-immune connections rather than flag the contradiction. That said, the false premise alone might be too easy to spot if the error is obvious, so it would be stronger if the premise were plausible enough to tempt a confident wrong answer. The brain does interact with the immune system — glial cells, brain-resident macrophages — so there's enough surface plausibility to make the false premise genuinely tricky rather than trivially wrong.",
  "chosen_strategy": "Embed a false premise — the question falsely implies the brain encodes immune memory.",
  "updated_question": "What brain regions encode immune memory?",
  "why_harder": "A shallow system may attempt to answer the question as asked, hallucinating a connection between brain regions and immune memory.",
  "verification_criterion": "The answer must explicitly flag the false premise (brain ≠ immune memory site) and correctly identify the actual biological structures involved before providing any substantive response."
}}

Seed question: Tell me about rubric generation models.
Output:
{{
  "brainstorming": "I could ask for a survey of rubric generation models, but that's basically the original question restated — a deep research system would just retrieve a list and summarize it. What if instead we asked for a failure case, like a deployed system that produced biased assessments? That's harder, but the field is niche enough that a system might just confabulate a plausible-sounding example. I could ask about general-purpose LLMs prompted to generate rubrics versus models actually trained for this task, and a shallow system will likely fail to distinguish between them.",
  "chosen_strategy": "Narrow scope to force multi-step reasoning — restricting to models purpose-trained for rubric generation requires distinguishing architectural and training choices rather than describing general LLM prompting workflows.",
  "updated_question": "Tell me about models trained specifically for rubric generation.",
  "why_harder": "General LLM prompting for rubric generation is well-documented and easy to summarize. Models purpose-trained for this task are rare, poorly documented, and require the responder to reason about methodology.",
  "verification_criterion": "For the models/papers mentioned, the answer should correctly identify whether there was training, or if the model was only prompted."
}}

Seed question: {seed_question}
"""


def load_seed_questions_from_file(input_file):
    """Load seed questions from text file (one per line)."""
    questions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(line)
    return questions

def load_seed_questions_from_hf(dataset_name="akariasai/openscholar_source_input_gpt5_s2snippets", split="train"):
    """Load seed questions from the 'question' column of a HuggingFace dataset."""
    dataset = load_dataset(dataset_name, split=split)
    return [row["question"] for row in dataset if row["question"]]

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
    
    # Default prompt template: use current_prompt formatted for qwen3-8b
    if prompt_template is None:
        def default_prompt_template(question):
            return [
                {"role": "user", "content": current_prompt.format(seed_question=question)}
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
  # From text file
  python convert_data_for_verl.py --input data/seed_questions.txt --output data/seed_questions.parquet

  # From HuggingFace dataset (default)
  python convert_data_for_verl.py --hf_dataset akariasai/openscholar_source_input_gpt5_s2snippets --output data/seed_questions.parquet
        """
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=str,
        help="Input text file with seed questions (one per line)"
    )
    source.add_argument(
        "--hf_dataset",
        type=str,
        default="akariasai/openscholar_source_input_gpt5_s2snippets",
        help="HuggingFace dataset name to load questions from"
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

    # Load questions from the appropriate source
    if args.input:
        if not Path(args.input).exists():
            print(f"ERROR: Input file not found: {args.input}")
            return 1
        print(f"Loading questions from {args.input}...")
        questions = load_seed_questions_from_file(args.input)
        source_label = args.input
    else:
        print(f"Loading questions from HuggingFace dataset '{args.hf_dataset}'")
        questions = load_seed_questions_from_hf(args.hf_dataset)
        source_label = f"{args.hf_dataset}"
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
    print(f"Input:  {source_label} ({len(questions)} questions)")
    print(f"Output: {args.output}")
    print(f"\nNext steps:")
    print(f"1. Update verl_config.yaml to point to: {args.output}")
    print(f"2. Run: python train_verl.py --config configs/verl_config.yaml")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())