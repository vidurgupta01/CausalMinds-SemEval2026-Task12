#!/usr/bin/env python3
"""
Simple multi-model ensemble: Run same baseline prompt on multiple models, vote on results.
No complex multi-stage - just the baseline that works, with voting.
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import sys

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from src.llm_engine import LLMEngine


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def baseline_prompt(q):
    """The simple prompt that actually works."""
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


def ensemble_vote(predictions: list[set[str]], threshold=0.5) -> set[str]:
    """Vote across predictions. Include option if >= threshold models agree."""
    vote_counts = Counter()
    for pred in predictions:
        for letter in pred:
            vote_counts[letter] += 1

    n = len(predictions)
    result = {letter for letter, count in vote_counts.items() if count >= threshold * n}

    if not result and vote_counts:
        result = {vote_counts.most_common(1)[0][0]}

    return result if result else {"A"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="sample")
    parser.add_argument("--models", default="gpt-4o,claude-sonnet-4-20250514",
                       help="Comma-separated model names")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Vote threshold (0.5 = majority)")
    args = parser.parse_args()

    questions = load_questions(args.split)
    if args.max_samples:
        questions = questions[:args.max_samples]

    model_names = [m.strip() for m in args.models.split(",")]
    engines = {name: LLMEngine.from_model_name(name) for name in model_names}

    print(f"Running SIMPLE MULTI-MODEL ENSEMBLE on {len(questions)} samples")
    print(f"Models: {model_names}")
    print(f"Vote threshold: {args.threshold}")
    print()

    scores_list = []
    results = []

    for q in tqdm(questions):
        prompt = baseline_prompt(q)
        predictions = []
        model_responses = {}

        for name, engine in engines.items():
            response = engine.get_response(prompt, max_tokens=64)
            pred = parse_response(response)
            predictions.append(pred)
            model_responses[name] = {
                "response": response[:100],
                "prediction": ",".join(sorted(pred))
            }

        final_pred = ensemble_vote(predictions, args.threshold)
        gold = set(q["golden_answer"].split(","))
        s = score(final_pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "event": q["target_event"][:80],
            "model_responses": model_responses,
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
    print(f"SIMPLE ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Models: {model_names}")
    print(f"Samples: {len(questions)}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"{'='*60}")

    # Per-model breakdown
    print("\nPer-model accuracy (when used alone):")
    for i, name in enumerate(model_names):
        model_scores = []
        for r in results:
            pred = set(r["model_responses"][name]["prediction"].split(","))
            gold = set(r["gold"].split(","))
            model_scores.append(score(pred, gold))
        model_avg = sum(model_scores) / len(model_scores)
        print(f"  {name}: {model_avg:.4f}")

    # Save
    model_tag = "_".join(m.split("/")[-1].replace("-", "")[:10] for m in model_names)
    output = {
        "config": {
            "models": model_names,
            "method": "simple_ensemble",
            "threshold": args.threshold,
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

    outfile = f"experiments/simple_ensemble_{model_tag}_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
