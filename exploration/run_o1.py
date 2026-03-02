#!/usr/bin/env python3
"""
Run with OpenAI o1 reasoning model.
"""

import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from tqdm import tqdm


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_prompt(q):
    """Simple prompt for o1."""
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

Answer with the letter(s) of the direct cause(s), comma-separated if multiple:"""


def get_response(client, prompt, model="o1"):
    """Get o1 response."""
    import time
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=256,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep((attempt + 1) * 5)
            else:
                raise
    return "A"


def parse_response(response: str) -> set[str]:
    """Parse response to extract letter answers."""
    response = response.upper().strip()
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    # Find letters at end
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
    parser.add_argument("--model", default="o1", help="o1, o1-mini, o1-preview")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    client = OpenAI()
    questions = load_questions(args.split)

    if args.max_samples:
        questions = questions[:args.max_samples]

    print(f"Running {args.model.upper()} on {len(questions)} samples")
    print()

    scores_list = []
    results = []

    for q in tqdm(questions):
        prompt = make_prompt(q)
        response = get_response(client, prompt, model=args.model)
        pred = parse_response(response)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "response": response[:200],
            "score": s,
        })

    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"{args.model.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {len(questions)}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"{'='*60}")

    # Save
    output = {
        "config": {"model": args.model, "split": args.split, "num_samples": len(questions)},
        "metrics": {"average_score": avg, "full_match": full, "partial_match": partial, "incorrect": wrong},
        "predictions": results,
    }

    outfile = f"experiments/{args.model}_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
