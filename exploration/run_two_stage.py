#!/usr/bin/env python3
"""
Two-stage approach to address under-prediction:
1. First pass: Get initial causes (baseline-style)
2. Second pass: Only for single-answer predictions, check if any other option could also be a cause
This avoids the over-prediction problem of multi-aware prompts.
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


def load_data(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return questions


def stage1_get_causes(question):
    """Stage 1: Get initial causes (baseline approach)."""
    prompt = f"""What directly CAUSED this event? A cause must happen BEFORE the event.

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Answer with letter(s) only, comma-separated if multiple:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def stage2_check_additional(question, initial_causes):
    """Stage 2: Check if any other option could also be a cause."""
    # Get remaining options
    all_options = {'A', 'B', 'C', 'D'}
    remaining = all_options - initial_causes

    if not remaining:
        return set()  # Already selected all

    # Format remaining options
    options_text = []
    for opt in sorted(remaining):
        opt_text = question[f'option_{opt}']
        options_text.append(f"{opt}. {opt_text}")

    prompt = f"""You identified "{', '.join(sorted(initial_causes))}" as cause(s) of: {question['target_event']}

Review these OTHER options - could any of them ALSO be a direct cause?

{chr(10).join(options_text)}

IMPORTANT: Only select if it clearly happened BEFORE the event and directly caused it.
Most events have only 1-2 causes. Be selective.

Additional cause(s)? Answer with letter(s) or "none":"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str) -> set[str]:
    response = response.upper().strip()
    if response == "NONE" or "NONE" in response:
        return set()
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    found = set(re.findall(r'\b([A-D])\b', response[-50:]))
    return found if found else set()


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
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    questions = load_data(args.split)
    questions = questions[:args.max_samples]

    print(f"Running TWO-STAGE on {len(questions)} samples\n")

    scores_list = []
    results = []
    stage2_triggered = 0

    for q in tqdm(questions):
        # Stage 1
        response1 = stage1_get_causes(q)
        initial = parse_response(response1)
        if not initial:
            initial = {"A"}  # Fallback

        final = initial.copy()

        # Stage 2: Only if single answer, check for additional
        if len(initial) == 1:
            response2 = stage2_check_additional(q, initial)
            additional = parse_response(response2)
            if additional:
                final = initial | additional
                stage2_triggered += 1

        gold = set(q["golden_answer"].split(","))
        s = score(final, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "stage1": ",".join(sorted(initial)),
            "final": ",".join(sorted(final)),
            "gold": q["golden_answer"],
            "score": s,
        })

    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"TWO-STAGE RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial:    {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong:      {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"Stage 2 triggered: {stage2_triggered} times")
    print(f"Baseline: 0.6825")
    print(f"{'='*60}")

    with open(f"experiments/two_stage_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
