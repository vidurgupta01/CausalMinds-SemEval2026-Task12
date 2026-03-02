#!/usr/bin/env python3
"""
Self-consistency approach: Run the same prompt N times with temperature,
then aggregate votes.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
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


def make_prompt(q):
    """Same prompt as baseline, but we'll run it multiple times."""
    return f"""Identify the direct cause(s) of this event.

Event: {q['target_event']}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

A direct cause must:
1. Happen BEFORE the event
2. Have a clear causal mechanism that leads to the event
3. Be necessary for the event to occur

Answer with the letter(s) of the direct cause(s), comma-separated if multiple:"""


def parse_response(response: str) -> set[str]:
    """Parse response to extract letter answers."""
    response = response.upper().strip()
    # Try exact match first
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    # Find any letters
    found = set(re.findall(r'\b([A-D])\b', response[-50:]))
    return found if found else {"A"}


def aggregate_votes(predictions: list[set[str]], threshold: float = 0.5) -> set[str]:
    """
    Aggregate multiple predictions using voting.
    Include an option if it appears in >= threshold fraction of predictions.
    """
    vote_counts = Counter()
    for pred in predictions:
        for label in pred:
            vote_counts[label] += 1

    n = len(predictions)
    result = {label for label, count in vote_counts.items() if count >= threshold * n}

    # If nothing passes threshold, take the most common
    if not result and vote_counts:
        result = {vote_counts.most_common(1)[0][0]}

    return result if result else {"A"}


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
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per question")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--threshold", type=float, default=0.4, help="Vote threshold (lower = more inclusive)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions(args.split)
    if args.max_samples:
        questions = questions[:args.max_samples]

    print(f"Running SELF-CONSISTENCY on {len(questions)} samples")
    print(f"Model: {args.model}, Samples: {args.samples}, Temp: {args.temp}, Threshold: {args.threshold}")
    print()

    # Initialize LLM engine
    engine = LLMEngine.from_model_name(args.model)
    scores_list = []
    results = []

    for q in tqdm(questions):
        prompt = make_prompt(q)

        # Get multiple samples
        predictions = []
        for _ in range(args.samples):
            response = engine.get_response(prompt, temperature=args.temp)
            pred = parse_response(response)
            predictions.append(pred)

        # Aggregate
        final_pred = aggregate_votes(predictions, args.threshold)
        gold = set(q["golden_answer"].split(","))
        s = score(final_pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "individual_preds": [",".join(sorted(p)) for p in predictions],
            "final_pred": ",".join(sorted(final_pred)),
            "gold": q["golden_answer"],
            "score": s,
        })

    # Print results
    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"SELF-CONSISTENCY RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(questions)}")
    print(f"Settings: {args.samples} samples/question, temp={args.temp}, threshold={args.threshold}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"{'='*60}")

    # Save
    output = {
        "config": {
            "model": args.model,
            "method": "self_consistency",
            "split": args.split,
            "samples_per_question": args.samples,
            "temperature": args.temp,
            "vote_threshold": args.threshold,
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

    outfile = f"experiments/{args.model}_self_consistency_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
