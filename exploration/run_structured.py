#!/usr/bin/env python3
"""
Structured output approach: Use JSON mode for reliable multi-label classification.
Based on OpenAI's structured outputs documentation.
"""

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


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def structured_eval(question):
    """Use structured JSON output for evaluation."""
    prompt = f"""Analyze this event and determine which options are direct CAUSES.
A direct cause must: (1) happen BEFORE the event, (2) directly lead to the event.

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Return a JSON object with:
- "analysis": Brief reasoning for each option
- "causes": Array of letters (A/B/C/D) that are direct causes

Example format:
{{"analysis": {{"A": "reason", "B": "reason", "C": "reason", "D": "reason"}}, "causes": ["A", "C"]}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str) -> set[str]:
    try:
        data = json.loads(response)
        causes = data.get("causes", [])
        if causes:
            return set(c.upper() for c in causes if c.upper() in ['A', 'B', 'C', 'D'])
    except:
        pass

    # Fallback parsing
    found = set(re.findall(r'\b([A-D])\b', response.upper()[-100:]))
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
    parser.add_argument("--split", default="sample")
    parser.add_argument("--max-samples", type=int, default=30)
    args = parser.parse_args()

    questions = load_questions(args.split)[:args.max_samples]

    print(f"Running STRUCTURED JSON OUTPUT on {len(questions)} samples\n")

    scores_list = []
    results = []

    for q in tqdm(questions):
        response = structured_eval(q)
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

    print(f"\n{'='*60}")
    print(f"STRUCTURED JSON OUTPUT RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Baseline: 0.6825")
    print(f"{'='*60}")

    with open(f"experiments/structured_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
