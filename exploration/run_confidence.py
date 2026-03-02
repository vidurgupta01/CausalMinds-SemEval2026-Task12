#!/usr/bin/env python3
"""
Confidence-based approach: Ask model for confidence score per option.
Select options above threshold as causes.
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


def get_confidence_scores(question):
    """Ask model for confidence score (0-100) for each option being a cause."""
    prompt = f"""For this event, rate how confident you are (0-100) that each option is a DIRECT CAUSE.
A direct cause must happen BEFORE the event and lead to it.

Event: {question['target_event']}

Rate each option's likelihood of being a direct cause:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Respond in exactly this format:
A: [score]
B: [score]
C: [score]
D: [score]"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def parse_confidence(response):
    """Parse confidence scores from response."""
    scores = {}
    for letter in ['A', 'B', 'C', 'D']:
        match = re.search(rf'{letter}[:\s]+(\d+)', response)
        if match:
            scores[letter] = int(match.group(1))
        else:
            scores[letter] = 0
    return scores


def select_causes(scores, threshold=60):
    """Select options with confidence above threshold."""
    selected = [letter for letter, score in scores.items() if score >= threshold]
    if not selected:
        # If nothing above threshold, pick highest
        selected = [max(scores, key=scores.get)]
    return set(selected)


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
    parser.add_argument("--threshold", type=int, default=60)
    args = parser.parse_args()

    questions = load_questions(args.split)[:args.max_samples]

    print(f"Running CONFIDENCE-BASED on {len(questions)} samples")
    print(f"Threshold: {args.threshold}\n")

    scores_list = []
    results = []

    for q in tqdm(questions):
        response = get_confidence_scores(q)
        conf_scores = parse_confidence(response)
        pred = select_causes(conf_scores, args.threshold)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
            "confidence": conf_scores
        })

    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)

    print(f"\n{'='*60}")
    print(f"CONFIDENCE-BASED RESULTS (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Baseline: 0.6825")
    print(f"{'='*60}")

    with open(f"experiments/confidence_t{args.threshold}_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "threshold": args.threshold, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
