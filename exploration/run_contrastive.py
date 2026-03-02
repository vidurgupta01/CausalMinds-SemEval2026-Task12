#!/usr/bin/env python3
"""
Contrastive prompting: Compare options pairwise to determine causes.
Based on research showing contrastive reasoning helps.
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


def contrastive_eval(question):
    """Use contrastive prompting to evaluate each option."""
    options = {
        'A': question['option_A'],
        'B': question['option_B'],
        'C': question['option_C'],
        'D': question['option_D']
    }

    prompt = f"""Event: {question['target_event']}

For each option, determine if it's a CAUSE or NOT A CAUSE of this event.
A cause must: (1) happen BEFORE the event, (2) directly lead to the event.

Think step by step for each:

A. {options['A']}
- Did this happen before the event?
- Did this directly cause the event?
- Classification: CAUSE or NOT_CAUSE?

B. {options['B']}
- Did this happen before the event?
- Did this directly cause the event?
- Classification: CAUSE or NOT_CAUSE?

C. {options['C']}
- Did this happen before the event?
- Did this directly cause the event?
- Classification: CAUSE or NOT_CAUSE?

D. {options['D']}
- Did this happen before the event?
- Did this directly cause the event?
- Classification: CAUSE or NOT_CAUSE?

Final answer - list only the letters that are CAUSES (comma-separated):"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str) -> set[str]:
    # Look for final answer line
    lines = response.split('\n')
    for line in reversed(lines):
        if 'final' in line.lower() or ':' in line:
            found = set(re.findall(r'\b([A-D])\b', line.upper()))
            if found:
                return found

    # Fallback: count CAUSE classifications
    causes = set()
    for letter in ['A', 'B', 'C', 'D']:
        pattern = rf'{letter}.*?Classification:\s*(CAUSE|NOT_CAUSE)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match and match.group(1).upper() == 'CAUSE':
            causes.add(letter)

    if causes:
        return causes

    # Last resort
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

    print(f"Running CONTRASTIVE PROMPTING on {len(questions)} samples\n")

    scores_list = []
    results = []

    for q in tqdm(questions):
        response = contrastive_eval(q)
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
    print(f"CONTRASTIVE PROMPTING RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Baseline: 0.6825")
    print(f"{'='*60}")

    with open(f"experiments/contrastive_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
