#!/usr/bin/env python3
"""Test the fine-tuned gpt-4.1-mini model."""

import json
import re
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

FINETUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal::D2UIiELC"


def load_data(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return questions


def run_finetuned(question):
    """Use the same prompt format as training."""
    prompt = f"""What directly CAUSED this event? A cause must happen BEFORE the event.

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Answer with letter(s) only, comma-separated if multiple:"""

    response = client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str) -> set[str]:
    response = response.upper().strip()
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    found = set(re.findall(r'\b([A-D])\b', response[-50:]))
    return found if found else {"A"}


def score(pred, gold):
    if pred == gold:
        return 1.0
    elif pred and pred.issubset(gold):
        return 0.5
    return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    questions = load_data(args.split)
    questions = questions[:args.max_samples]

    print(f"Testing FINE-TUNED MODEL on {len(questions)} {args.split} samples")
    print(f"Model: {FINETUNED_MODEL}\n")

    scores_list = []
    results = []

    for q in tqdm(questions):
        response = run_finetuned(q)
        pred = parse_response(response)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
        })

    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"FINE-TUNED MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial:    {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong:      {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"Baseline (GPT-4o): 0.6825")
    print(f"{'='*60}")

    with open(f"experiments/finetuned_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "model": FINETUNED_MODEL, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
