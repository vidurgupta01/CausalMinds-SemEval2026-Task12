#!/usr/bin/env python3
"""
Multi-answer aware approach: Use a prompt that explicitly considers
the possibility of multiple correct answers.
"""

import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from src.llm_engine import LLMEngine


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_multi_aware_prompt(q):
    """Prompt that explicitly considers multiple causes."""
    return f"""Identify ALL direct causes of this event. Multiple options can be correct.

Event: {q['target_event']}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

For EACH option, consider:
1. Did this happen BEFORE the event? (If after, it's an EFFECT, not a cause)
2. Is there a clear causal mechanism?
3. Would the event have happened without this?

Many events have MULTIPLE causes working together. Include ALL options that are direct causes.

Answer with all letters that are direct causes (comma-separated):"""


def parse_response(response: str) -> set[str]:
    """Parse response to extract letter answers."""
    response = response.upper().strip()
    # Try exact match first
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    # Find any letters in last part
    found = set(re.findall(r'\b([A-D])\b', response[-100:]))
    return found if found else {"A"}


def score(pred, gold):
    if pred == gold:
        return 1.0
    elif pred and pred.issubset(gold):
        return 0.5
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="sample")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions(args.split)
    if args.max_samples:
        questions = questions[:args.max_samples]

    print(f"Running MULTI-AWARE approach on {len(questions)} samples")
    print(f"Model: {args.model}")
    print()

    engine = LLMEngine.from_model_name(args.model)
    scores_list = []
    results = []

    for q in tqdm(questions):
        prompt = make_multi_aware_prompt(q)
        response = engine.get_response(prompt, max_tokens=64)
        pred = parse_response(response)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
            "response": response[:100],
        })

    # Print results
    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"MULTI-AWARE RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(questions)}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"{'='*60}")

    # Analyze by gold answer count
    multi_gold = [r for r in results if len(r["gold"].split(",")) > 1]
    single_gold = [r for r in results if len(r["gold"].split(",")) == 1]

    if multi_gold:
        multi_avg = sum(r["score"] for r in multi_gold) / len(multi_gold)
        print(f"\nMulti-answer ({len(multi_gold)}): {multi_avg:.4f}")
    if single_gold:
        single_avg = sum(r["score"] for r in single_gold) / len(single_gold)
        print(f"Single-answer ({len(single_gold)}): {single_avg:.4f}")

    # Save
    model_short = args.model.split("/")[-1]
    output = {
        "config": {
            "model": args.model,
            "method": "multi_aware",
            "split": args.split,
            "num_samples": len(questions),
        },
        "metrics": {
            "average_score": avg,
            "full_match": full,
            "partial_match": partial,
            "incorrect": wrong,
            "total": len(questions),
        },
        "predictions": results,
    }

    outfile = f"experiments/{model_short}_multi_aware_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
