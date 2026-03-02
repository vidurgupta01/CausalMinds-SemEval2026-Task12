#!/usr/bin/env python3
"""Test focused causation prompt."""

import json
import random
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from tqdm import tqdm

# Load data
with open("data/official/sample_data/questions.jsonl") as f:
    questions = [json.loads(line) for line in f if line.strip()][:30]

with open("data/official/train_data/questions.jsonl") as f:
    train = [json.loads(line) for line in f if line.strip()]

# Get good examples
random.seed(42)
examples = random.sample(train, 5)


def make_prompt(q, examples):
    ex_text = "\n\n".join([
        f"Event: {e['target_event']}\nA. {e['option_A']}\nB. {e['option_B']}\nC. {e['option_C']}\nD. {e['option_D']}\nAnswer: {e['golden_answer']}"
        for e in examples
    ])

    return f"""You must identify what DIRECTLY CAUSED an event. Not what is related, not what happened after - only what CAUSED it.

CRITICAL DISTINCTIONS:
- CAUSE = happens BEFORE and MAKES the event happen
- CORRELATION = related but doesn't cause
- EFFECT = happens AFTER (cannot be a cause!)
- "None of the others" = valid when no option is a true direct cause

Examples:
{ex_text}

---

Event: {q['target_event']}
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

Analyze each briefly:
- A: Before/after event? Direct cause or just related?
- B: Before/after event? Direct cause or just related?
- C: Before/after event? Direct cause or just related?
- D: Before/after event? Direct cause or just related?

Final answer (letters only, comma-separated if multiple):"""


client = OpenAI()


def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content


def parse(response):
    import re
    response = response.upper()
    lines = response.strip().split('\n')
    for line in reversed(lines):
        match = re.search(r'^([A-D](?:\s*,\s*[A-D])*)$', line.strip())
        if match:
            return set(match.group(1).replace(" ", "").split(","))
    # Fallback: find any letters
    match = re.search(r'([A-D](?:\s*,\s*[A-D])*)\s*$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    return {"A"}


def score(pred, gold):
    if pred == gold:
        return 1.0
    elif pred and pred.issubset(gold):
        return 0.5
    return 0.0


print("Running focused causation prompt on 30 samples...")
scores = []
results = []

for q in tqdm(questions):
    prompt = make_prompt(q, examples)
    response = get_response(prompt)
    pred = parse(response)
    gold = set(q["golden_answer"].split(","))
    s = score(pred, gold)
    scores.append(s)
    results.append({
        "id": q["id"],
        "event": q["target_event"],
        "gold": q["golden_answer"],
        "pred": ",".join(sorted(pred)),
        "score": s,
    })

avg = sum(scores) / len(scores)
full = sum(1 for s in scores if s == 1.0)
partial = sum(1 for s in scores if s == 0.5)
wrong = sum(1 for s in scores if s == 0.0)

print(f"\n{'='*60}")
print(f"FOCUSED CAUSATION PROMPT RESULTS")
print(f"{'='*60}")
print(f"Average Score: {avg:.4f}")
print(f"Full Match: {full}/30 ({full/30*100:.1f}%)")
print(f"Partial: {partial}/30 ({partial/30*100:.1f}%)")
print(f"Wrong: {wrong}/30 ({wrong/30*100:.1f}%)")
print(f"{'='*60}")

# Save results
with open("experiments/gpt-4o_focused_causation_sample_results.json", "w") as f:
    json.dump({"scores": {"avg": avg, "full": full, "partial": partial, "wrong": wrong}, "results": results}, f, indent=2)
