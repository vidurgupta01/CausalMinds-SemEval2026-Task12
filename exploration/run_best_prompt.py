#!/usr/bin/env python3
"""
Run the best prompt variant (v2_temporal) on full dev set.
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


def temporal_prompt(q):
    """Best performing prompt variant - emphasizes temporal order."""
    return f"""Identify what CAUSED this event. A cause must happen BEFORE the event.

Event: {q['target_event']}

A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

Which happened BEFORE and CAUSED the event? (letters only, comma-separated if multiple):"""


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions(args.split)
    if args.max_samples:
        questions = questions[:args.max_samples]

    engine = LLMEngine.from_model_name(args.model)

    print(f"Running TEMPORAL PROMPT (v2) on {len(questions)} samples")
    print(f"Model: {args.model}, Split: {args.split}")
    print()

    scores_list = []
    results = []

    for q in tqdm(questions):
        prompt = temporal_prompt(q)
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
        })

    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"TEMPORAL PROMPT RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split} ({len(questions)} samples)")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\nBaseline dev score was 0.6825 - compare to see if this helps!")

    # Save
    output = {
        "config": {"model": args.model, "prompt": "v2_temporal", "split": args.split},
        "metrics": {"average_score": avg, "full_match": full, "partial_match": partial, "incorrect": wrong},
        "predictions": results,
    }

    outfile = f"experiments/{args.model}_temporal_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
