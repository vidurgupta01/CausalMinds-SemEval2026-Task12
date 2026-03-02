#!/usr/bin/env python3
"""
Per-option evaluation: Ask "Is this a cause?" for each option separately.
Designed to catch multi-answer cases the model misses.
"""

import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import re
import sys

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from src.llm_engine import LLMEngine


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_per_option_prompt(q, letter, option):
    """Ask about one specific option."""
    return f"""Event: {q['target_event']}

Potential cause: {option}

Question: Did this potential cause DIRECTLY cause the event?

A direct cause must:
1. Happen BEFORE the event (temporal order)
2. Have a clear mechanism that leads to the event
3. If removed, the event would NOT have happened

Answer YES or NO only:"""


def parse_yes_no(response: str) -> bool:
    """Parse YES/NO response."""
    response = response.upper().strip()
    return "YES" in response


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

    print(f"Running PER-OPTION evaluation on {len(questions)} samples")
    print(f"Model: {args.model}")
    print()

    # Initialize LLM engine
    engine = LLMEngine.from_model_name(args.model)
    scores_list = []
    results = []

    for q in tqdm(questions):
        option_responses = {}
        pred = set()

        # Evaluate each option independently
        for letter in "ABCD":
            option = q[f"option_{letter}"]
            prompt = make_per_option_prompt(q, letter, option)
            response = engine.get_response(prompt, max_tokens=16)
            is_cause = parse_yes_no(response)
            option_responses[letter] = {"response": response, "is_cause": is_cause}
            if is_cause:
                pred.add(letter)

        # If nothing was marked as cause, default to most likely
        if not pred:
            pred = {"A"}

        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "option_responses": option_responses,
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
        })

    # Print results
    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"PER-OPTION EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(questions)}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"{'='*60}")

    # Analyze prediction patterns
    num_causes = [len(set(r["pred"].split(","))) for r in results]
    from collections import Counter
    print(f"\nPrediction size distribution: {Counter(num_causes)}")

    # Save
    output = {
        "config": {
            "model": args.model,
            "method": "per_option",
            "split": args.split,
            "num_samples": len(questions),
        },
        "metrics": {
            "average_score": avg,
            "full_match": full,
            "partial_match": partial,
            "incorrect": wrong,
            "total": len(questions),
            "full_match_rate": full/len(questions),
            "partial_match_rate": partial/len(questions),
            "incorrect_rate": wrong/len(questions),
        },
        "predictions": results,
    }

    outfile = f"experiments/{args.model}_per_option_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
