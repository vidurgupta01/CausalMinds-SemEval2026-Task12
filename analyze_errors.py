#!/usr/bin/env python3
"""Analyze error patterns in baseline results."""

import json
from collections import Counter

# Load baseline results
with open("experiments/gpt-4o_zero_shot_no_docs_dev_results.json") as f:
    data = json.load(f)

# Load questions for full context
with open("data/official/dev_data/questions.jsonl") as f:
    questions = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

predictions = data["predictions"]

# Categorize errors
missing_causes = []  # Predicted subset of gold (partial = 0.5)
false_positives = []  # Predicted superset of gold or unrelated
wrong = []  # Completely wrong

for p in predictions:
    pred_set = set(p["prediction"].split(",")) if p["prediction"] else set()
    gold_set = set(p["gold"].split(","))

    if pred_set == gold_set:
        continue  # Correct

    if pred_set.issubset(gold_set):
        missing_causes.append(p)
    elif gold_set.issubset(pred_set):
        false_positives.append(p)
    else:
        wrong.append(p)

print(f"Error Analysis")
print(f"="*60)
print(f"Total samples: {len(predictions)}")
print(f"Correct: {len(predictions) - len(missing_causes) - len(false_positives) - len(wrong)}")
print(f"Missing causes (partial): {len(missing_causes)}")
print(f"False positives: {len(false_positives)}")
print(f"Wrong (mixed errors): {len(wrong)}")

# Analyze how many causes we miss vs add
print(f"\n{'='*60}")
print("MISSING CAUSES - Model predicts fewer than gold")
print(f"{'='*60}")

for i, p in enumerate(missing_causes[:5]):
    q = questions.get(p["id"], {})
    print(f"\n[{i+1}] {p['id']}")
    print(f"Event: {p['target_event'][:80]}...")
    print(f"Gold: {p['gold']} | Pred: {p['prediction']}")
    missing = set(p["gold"].split(",")) - set(p["prediction"].split(","))
    print(f"Missed: {missing}")
    for letter in missing:
        opt_key = f"option_{letter}"
        if opt_key in q:
            print(f"  {letter}: {q[opt_key][:80]}...")

print(f"\n{'='*60}")
print("FALSE POSITIVES - Model predicts more than gold")
print(f"{'='*60}")

for i, p in enumerate(false_positives[:5]):
    q = questions.get(p["id"], {})
    print(f"\n[{i+1}] {p['id']}")
    print(f"Event: {p['target_event'][:80]}...")
    print(f"Gold: {p['gold']} | Pred: {p['prediction']}")
    extra = set(p["prediction"].split(",")) - set(p["gold"].split(","))
    print(f"Extra: {extra}")
    for letter in extra:
        opt_key = f"option_{letter}"
        if opt_key in q:
            print(f"  {letter}: {q[opt_key][:80]}...")

print(f"\n{'='*60}")
print("WRONG - Mixed errors")
print(f"{'='*60}")

for i, p in enumerate(wrong[:5]):
    q = questions.get(p["id"], {})
    print(f"\n[{i+1}] {p['id']}")
    print(f"Event: {p['target_event'][:80]}...")
    print(f"Gold: {p['gold']} | Pred: {p['prediction']}")
    missing = set(p["gold"].split(",")) - set(p["prediction"].split(","))
    extra = set(p["prediction"].split(",")) - set(p["gold"].split(","))
    print(f"Missed: {missing}, Extra: {extra}")

# Distribution analysis
print(f"\n{'='*60}")
print("GOLD ANSWER DISTRIBUTION IN ERRORS")
print(f"{'='*60}")

all_errors = missing_causes + false_positives + wrong
gold_sizes = Counter(len(p["gold"].split(",")) for p in all_errors)
pred_sizes = Counter(len(p["prediction"].split(",")) for p in all_errors)

print(f"Gold answer sizes: {dict(gold_sizes)}")
print(f"Predicted sizes: {dict(pred_sizes)}")

# Multi-answer analysis
multi_gold = [p for p in predictions if len(p["gold"].split(",")) > 1]
multi_wrong = [p for p in all_errors if len(p["gold"].split(",")) > 1]
print(f"\nMulti-answer questions: {len(multi_gold)}")
print(f"Multi-answer errors: {len(multi_wrong)} ({len(multi_wrong)/len(multi_gold)*100:.1f}% error rate)")

single_gold = [p for p in predictions if len(p["gold"].split(",")) == 1]
single_wrong = [p for p in all_errors if len(p["gold"].split(",")) == 1]
print(f"Single-answer questions: {len(single_gold)}")
print(f"Single-answer errors: {len(single_wrong)} ({len(single_wrong)/len(single_gold)*100:.1f}% error rate)")
