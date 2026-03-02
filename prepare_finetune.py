#!/usr/bin/env python3
"""
Prepare training data for fine-tuning.
Supports OpenAI format and HuggingFace format.
"""

import json
import random
from pathlib import Path


def load_questions(split: str) -> list[dict]:
    """Load questions from a split."""
    filepath = Path(f"data/official/{split}_data/questions.jsonl")
    questions = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def format_prompt(q: dict) -> str:
    """Format a question as a prompt."""
    return f"""Identify the direct cause(s) of this event. Select all options that directly caused the event to happen.

Event: {q['target_event']}

A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

Rules:
- A direct cause must happen BEFORE the event
- There must be a clear causal mechanism
- Multiple options can be correct
- "None of the others" is valid when no option is a true cause

Answer (letters only, comma-separated if multiple):"""


def format_openai_finetune(questions: list[dict]) -> list[dict]:
    """Format data for OpenAI fine-tuning."""
    formatted = []
    for q in questions:
        formatted.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at causal reasoning. Identify direct causes of events accurately."
                },
                {
                    "role": "user",
                    "content": format_prompt(q)
                },
                {
                    "role": "assistant",
                    "content": q["golden_answer"]
                }
            ]
        })
    return formatted


def format_huggingface(questions: list[dict]) -> list[dict]:
    """Format data for HuggingFace fine-tuning."""
    formatted = []
    for q in questions:
        formatted.append({
            "instruction": format_prompt(q),
            "output": q["golden_answer"],
            "input": "",  # No additional input
        })
    return formatted


def main():
    print("Loading training data...")
    train = load_questions("train")
    dev = load_questions("dev")

    print(f"Train: {len(train)} examples")
    print(f"Dev: {len(dev)} examples")

    # Shuffle training data
    random.seed(42)
    random.shuffle(train)

    # Split into train/val for fine-tuning (90/10)
    split_idx = int(len(train) * 0.9)
    train_ft = train[:split_idx]
    val_ft = train[split_idx:]

    print(f"\nFine-tuning split: {len(train_ft)} train, {len(val_ft)} val")

    # Create output directory
    output_dir = Path("data/finetune")
    output_dir.mkdir(exist_ok=True)

    # OpenAI format
    print("\nPreparing OpenAI format...")
    openai_train = format_openai_finetune(train_ft)
    openai_val = format_openai_finetune(val_ft)

    with open(output_dir / "openai_train.jsonl", "w") as f:
        for item in openai_train:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "openai_val.jsonl", "w") as f:
        for item in openai_val:
            f.write(json.dumps(item) + "\n")

    print(f"  Saved: {output_dir}/openai_train.jsonl ({len(openai_train)} examples)")
    print(f"  Saved: {output_dir}/openai_val.jsonl ({len(openai_val)} examples)")

    # HuggingFace format
    print("\nPreparing HuggingFace format...")
    hf_train = format_huggingface(train_ft)
    hf_val = format_huggingface(val_ft)

    with open(output_dir / "hf_train.json", "w") as f:
        json.dump(hf_train, f, indent=2)

    with open(output_dir / "hf_val.json", "w") as f:
        json.dump(hf_val, f, indent=2)

    print(f"  Saved: {output_dir}/hf_train.json ({len(hf_train)} examples)")
    print(f"  Saved: {output_dir}/hf_val.json ({len(hf_val)} examples)")

    # Also prepare full training set (for final model)
    print("\nPreparing full training set...")
    openai_full = format_openai_finetune(train)
    with open(output_dir / "openai_full.jsonl", "w") as f:
        for item in openai_full:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved: {output_dir}/openai_full.jsonl ({len(openai_full)} examples)")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("""
1. OpenAI Fine-tuning:
   openai api fine_tuning.jobs.create \\
     -t data/finetune/openai_train.jsonl \\
     -v data/finetune/openai_val.jsonl \\
     -m gpt-4o-mini-2024-07-18

2. Or use Python:
   from openai import OpenAI
   client = OpenAI()

   # Upload files
   train_file = client.files.create(
       file=open("data/finetune/openai_train.jsonl", "rb"),
       purpose="fine-tune"
   )

   # Create fine-tuning job
   client.fine_tuning.jobs.create(
       training_file=train_file.id,
       model="gpt-4o-mini-2024-07-18"
   )

3. For local models (Llama/Mistral), use the HuggingFace format files
   with transformers + PEFT/LoRA
""")


if __name__ == "__main__":
    main()
