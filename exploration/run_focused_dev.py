#!/usr/bin/env python3
"""Run focused causation prompt on full dev set."""

import json
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from tqdm import tqdm

# Load data
with open("data/official/dev_data/questions.jsonl") as f:
    questions = [json.loads(line) for line in f if line.strip()]

with open("data/official/train_data/questions.jsonl") as f:
    train = [json.loads(line) for line in f if line.strip()]

random.seed(42)
examples = random.sample(train, 5)


def make_prompt(q, examples):
    """Simple, focused prompt - no over-engineering."""
    ex_text = "\n\n".join([
        f"Event: {e['target_event']}\nA. {e['option_A']}\nB. {e['option_B']}\nC. {e['option_C']}\nD. {e['option_D']}\nAnswer: {e['golden_answer']}"
        for e in examples
    ])

    return f"""Identify what DIRECTLY CAUSED an event. Not what is related, not what happened after - only what CAUSED it.

CAUSE = happens BEFORE and MAKES the event happen
EFFECT = happens AFTER (cannot be a cause)
Multiple answers may be correct.

Examples:
{ex_text}

---

Event: {q['target_event']}
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

For each, ask: Did this happen BEFORE the event? Did it CAUSE the event?

Answer (letters only):"""


client = OpenAI()


def get_response(prompt, retries=3):
    import time
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 2
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")


def parse(response):
    import re
    response = response.upper()
    lines = response.strip().split('\n')
    for line in reversed(lines):
        match = re.search(r'^([A-D](?:\s*,\s*[A-D])*)$', line.strip())
        if match:
            return set(match.group(1).replace(" ", "").split(","))
    match = re.search(r'([A-D](?:\s*,\s*[A-D])*)\s*$', response)
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


print(f"Running FOCUSED PROMPT on DEV SET ({len(questions)} samples)...")
print("This is the real evaluation - no peeking at answers!\n")

# Try to resume from checkpoint
checkpoint_file = Path("experiments/focused_dev_checkpoint.json")
scores = []
results = []
start_idx = 0

if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
    results = checkpoint["results"]
    scores = checkpoint["scores"]
    start_idx = len(results)
    print(f"Resuming from checkpoint at {start_idx}/{len(questions)}")

for i, q in enumerate(tqdm(questions[start_idx:], initial=start_idx, total=len(questions))):
    prompt = make_prompt(q, examples)
    response = get_response(prompt)
    pred = parse(response)
    gold = set(q["golden_answer"].split(","))
    s = score(pred, gold)
    scores.append(s)
    results.append({
        "id": q["id"],
        "event": q["target_event"][:50],
        "gold": q["golden_answer"],
        "pred": ",".join(sorted(pred)),
        "score": s,
    })

    # Save checkpoint every 50 samples
    if len(results) % 50 == 0:
        with open(checkpoint_file, "w") as f:
            json.dump({"results": results, "scores": scores}, f)
        print(f" [checkpoint saved at {len(results)}]")

avg = sum(scores) / len(scores)
full = sum(1 for s in scores if s == 1.0)
partial = sum(1 for s in scores if s == 0.5)
wrong = sum(1 for s in scores if s == 0.0)

print(f"\n{'='*60}")
print(f"FOCUSED PROMPT - DEV SET RESULTS")
print(f"{'='*60}")
print(f"Samples: {len(questions)}")
print(f"Average Score: {avg:.4f}")
print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
print(f"{'='*60}")

# Save
output = {
    "config": {
        "model": "gpt-4o",
        "prompt": "focused_causation",
        "split": "dev",
        "n_examples": 5,
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

with open("experiments/gpt-4o_focused_causation_dev_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to experiments/gpt-4o_focused_causation_dev_results.json")
