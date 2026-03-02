#!/usr/bin/env python3
"""Test smart prompt v2 with discovered patterns."""

import json
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
from tqdm import tqdm


def has_none_option(q):
    """Check which option (if any) contains 'None of the others'."""
    for letter in ["A", "B", "C", "D"]:
        text = q[f"option_{letter}"].lower()
        if "none of the other" in text:
            return letter
    return None


def has_after_keyword(text):
    """Check if text suggests an effect (happens after)."""
    indicators = ["after", "following the", "resulted from", "in response to"]
    return any(ind in text.lower() for ind in indicators)


def smart_prompt_v2(q, examples):
    """Smart prompt with all discovered patterns."""

    none_opt = has_none_option(q)

    # Build option analysis hints
    hints = []
    for letter in ["A", "B", "C", "D"]:
        text = q[f"option_{letter}"]
        if has_after_keyword(text):
            hints.append(f"- Option {letter} contains 'after' - likely an EFFECT, not a cause")

    hints_text = "\n".join(hints) if hints else ""

    none_instruction = ""
    if none_opt:
        none_instruction = f"""
Note: Option {none_opt} claims "None of the others are correct causes."
Select {none_opt} ONLY IF you determine that options {"ABCD".replace(none_opt, "")} are all either:
- Effects (happen AFTER the event)
- Unrelated events
- Correlated but not causal"""

    examples_text = "\n\n".join([
        f"Event: {e['target_event']}\nA. {e['option_A']}\nB. {e['option_B']}\nC. {e['option_C']}\nD. {e['option_D']}\nAnswer: {e['golden_answer']}"
        for e in examples[:4]
    ])

    return f"""You are an expert at causal reasoning. Identify the DIRECT CAUSE(s) of an event.

KEY RULES:
1. CAUSE = happens BEFORE the event and MAKES it happen
2. EFFECT = happens AFTER the event (CANNOT be a cause!)
3. Words like "after", "following", "resulted from" indicate EFFECTS
4. ~43% of questions have MULTIPLE correct answers (2-3 options)
5. "None of the others are correct" is valid when other options aren't true causes
{none_instruction}

Examples:
{examples_text}

---

Event: {q['target_event']}

A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

{hints_text}

Analysis:
- A: Before or after event? Direct cause?
- B: Before or after event? Direct cause?
- C: Before or after event? Direct cause?
- D: Before or after event? Direct cause?

Answer (letters only, comma-separated if multiple):"""


# Load data
with open("data/official/sample_data/questions.jsonl") as f:
    questions = [json.loads(line) for line in f if line.strip()][:50]

with open("data/official/train_data/questions.jsonl") as f:
    train = [json.loads(line) for line in f if line.strip()]

random.seed(42)
examples = random.sample(train, 5)

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


print("Running SMART PROMPT V2 on 50 samples...")
scores = []
results = []

for q in tqdm(questions):
    prompt = smart_prompt_v2(q, examples)
    response = get_response(prompt)
    pred = parse(response)
    gold = set(q["golden_answer"].split(","))
    s = score(pred, gold)
    scores.append(s)
    results.append({
        "id": q["id"],
        "gold": q["golden_answer"],
        "pred": ",".join(sorted(pred)),
        "score": s,
    })

avg = sum(scores) / len(scores)
full = sum(1 for s in scores if s == 1.0)
partial = sum(1 for s in scores if s == 0.5)
wrong = sum(1 for s in scores if s == 0.0)

print(f"\n{'='*60}")
print(f"SMART PROMPT V2 RESULTS (50 samples)")
print(f"{'='*60}")
print(f"Average Score: {avg:.4f}")
print(f"Full Match: {full}/50 ({full/50*100:.1f}%)")
print(f"Partial: {partial}/50 ({partial/50*100:.1f}%)")
print(f"Wrong: {wrong}/50 ({wrong/50*100:.1f}%)")
print(f"{'='*60}")

# Save
with open("experiments/gpt-4o_smart_v2_sample_results.json", "w") as f:
    json.dump({"avg": avg, "full": full, "partial": partial, "wrong": wrong, "results": results}, f, indent=2)
