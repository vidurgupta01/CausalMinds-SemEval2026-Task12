#!/usr/bin/env python3
"""
Analyze errors and build a smart router based on question characteristics.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_results():
    """Load baseline dev results."""
    with open("experiments/gpt-4o_zero_shot_no_docs_dev_results.json") as f:
        return json.load(f)


def load_questions():
    """Load dev questions for full context."""
    with open("data/official/dev_data/questions.jsonl") as f:
        return {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}


def analyze_question(q):
    """Extract features from a question for routing."""
    event = q["target_event"].lower()
    options = [q[f"option_{l}"] for l in "ABCD"]
    options_lower = [o.lower() for o in options]

    features = {}

    # 1. Has "none of the others" option
    features["has_none_option"] = any("none of the other" in o for o in options_lower)

    # 2. Number of options mentioning dates
    date_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|\d{4}|\d{1,2}(st|nd|rd|th)?)\b'
    features["options_with_dates"] = sum(1 for o in options_lower if re.search(date_pattern, o))

    # 3. Event contains temporal words suggesting sequence
    temporal_after = ["after", "following", "then", "subsequently", "resulted in"]
    temporal_before = ["led to", "caused", "prompted", "triggered"]
    features["event_has_after_words"] = any(w in event for w in temporal_after)
    features["event_has_before_words"] = any(w in event for w in temporal_before)

    # 4. Topic domain detection
    domains = {
        "politics": ["president", "minister", "parliament", "congress", "vote", "election", "impeach", "law"],
        "economics": ["market", "stock", "bank", "tariff", "trade", "price", "billion", "trillion"],
        "tech": ["ai", "chatgpt", "meta", "twitter", "facebook", "nvidia", "huawei", "app"],
        "crisis": ["protest", "riot", "attack", "death", "killed", "war", "invasion", "emergency"],
        "legal": ["charged", "arrested", "trial", "court", "ruling", "verdict"],
    }

    text = event + " " + " ".join(options_lower)
    domain_scores = {}
    for domain, keywords in domains.items():
        domain_scores[domain] = sum(1 for k in keywords if k in text)
    features["primary_domain"] = max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else "general"

    # 5. Gold answer characteristics (from training)
    features["gold_count"] = len(q.get("golden_answer", "A").split(","))

    # 6. Option similarity to event (shared words)
    event_words = set(re.findall(r'\b[a-z]{4,}\b', event))
    option_overlaps = []
    for o in options_lower:
        opt_words = set(re.findall(r'\b[a-z]{4,}\b', o))
        overlap = len(event_words & opt_words)
        option_overlaps.append(overlap)
    features["max_option_overlap"] = max(option_overlaps)
    features["options_with_high_overlap"] = sum(1 for o in option_overlaps if o >= 2)

    return features


def main():
    results = load_results()
    questions = load_questions()
    predictions = results["predictions"]

    # Categorize by error type
    correct = []
    partial = []  # Predicted subset of gold
    over_predict = []  # Gold is subset of predicted
    wrong = []  # No overlap or wrong entirely

    for p in predictions:
        q = questions.get(p["id"], {})
        pred_set = set(p["prediction"].split(",")) if p["prediction"] else set()
        gold_set = set(p["gold"].split(","))

        p["features"] = analyze_question({**q, "golden_answer": p["gold"]})
        p["pred_set"] = pred_set
        p["gold_set"] = gold_set

        if pred_set == gold_set:
            correct.append(p)
        elif pred_set.issubset(gold_set):
            partial.append(p)
        elif gold_set.issubset(pred_set):
            over_predict.append(p)
        else:
            wrong.append(p)

    print("=" * 70)
    print("ERROR ANALYSIS FOR SMART ROUTING")
    print("=" * 70)
    print(f"Correct: {len(correct)} ({len(correct)/len(predictions)*100:.1f}%)")
    print(f"Partial (under-predict): {len(partial)} ({len(partial)/len(predictions)*100:.1f}%)")
    print(f"Over-predict: {len(over_predict)} ({len(over_predict)/len(predictions)*100:.1f}%)")
    print(f"Wrong: {len(wrong)} ({len(wrong)/len(predictions)*100:.1f}%)")

    # Analyze features by error type
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS BY ERROR TYPE")
    print("=" * 70)

    def analyze_group(group, name):
        print(f"\n### {name} (n={len(group)}) ###")
        if not group:
            return

        # Domain distribution
        domains = Counter(p["features"]["primary_domain"] for p in group)
        print(f"Domains: {dict(domains)}")

        # Gold count distribution
        gold_counts = Counter(p["features"]["gold_count"] for p in group)
        print(f"Gold answer counts: {dict(gold_counts)}")

        # Has none option
        has_none = sum(1 for p in group if p["features"]["has_none_option"])
        print(f"Has 'none' option: {has_none} ({has_none/len(group)*100:.1f}%)")

        # High overlap
        high_overlap = sum(1 for p in group if p["features"]["max_option_overlap"] >= 3)
        print(f"High option-event overlap: {high_overlap} ({high_overlap/len(group)*100:.1f}%)")

    analyze_group(correct, "CORRECT")
    analyze_group(partial, "PARTIAL (Under-predict)")
    analyze_group(over_predict, "OVER-PREDICT")
    analyze_group(wrong, "WRONG")

    # Find patterns for routing
    print("\n" + "=" * 70)
    print("ROUTING RECOMMENDATIONS")
    print("=" * 70)

    # Multi-answer questions
    multi_total = sum(1 for p in predictions if p["features"]["gold_count"] > 1)
    multi_correct = sum(1 for p in correct if p["features"]["gold_count"] > 1)
    multi_partial = sum(1 for p in partial if p["features"]["gold_count"] > 1)
    print(f"\nMulti-answer questions: {multi_total}")
    print(f"  Correct: {multi_correct} ({multi_correct/multi_total*100:.1f}%)")
    print(f"  Partial: {multi_partial} ({multi_partial/multi_total*100:.1f}%)")
    print(f"  Error rate: {(multi_total-multi_correct)/multi_total*100:.1f}%")

    # Single-answer questions
    single_total = sum(1 for p in predictions if p["features"]["gold_count"] == 1)
    single_correct = sum(1 for p in correct if p["features"]["gold_count"] == 1)
    print(f"\nSingle-answer questions: {single_total}")
    print(f"  Correct: {single_correct} ({single_correct/single_total*100:.1f}%)")
    print(f"  Error rate: {(single_total-single_correct)/single_total*100:.1f}%")

    # Questions with "none" option
    none_total = sum(1 for p in predictions if p["features"]["has_none_option"])
    none_correct = sum(1 for p in correct if p["features"]["has_none_option"])
    print(f"\nQuestions with 'none' option: {none_total}")
    print(f"  Correct: {none_correct} ({none_correct/none_total*100:.1f}%)")

    # Domain-specific accuracy
    print("\nAccuracy by domain:")
    for domain in ["politics", "economics", "tech", "crisis", "legal", "general"]:
        domain_total = sum(1 for p in predictions if p["features"]["primary_domain"] == domain)
        domain_correct = sum(1 for p in correct if p["features"]["primary_domain"] == domain)
        if domain_total > 0:
            print(f"  {domain}: {domain_correct}/{domain_total} ({domain_correct/domain_total*100:.1f}%)")

    # Print specific error examples
    print("\n" + "=" * 70)
    print("SAMPLE ERRORS FOR TARGETED PROMPTS")
    print("=" * 70)

    print("\n### UNDER-PREDICTIONS (Missed causes) ###")
    for p in partial[:5]:
        q = questions.get(p["id"], {})
        missed = p["gold_set"] - p["pred_set"]
        print(f"\nID: {p['id']}")
        print(f"Event: {p['target_event'][:70]}...")
        print(f"Gold: {p['gold']} | Pred: {p['prediction']} | Missed: {missed}")
        for letter in missed:
            print(f"  {letter}: {q.get(f'option_{letter}', 'N/A')[:60]}...")

    print("\n### OVER-PREDICTIONS (False positives) ###")
    for p in over_predict[:5]:
        q = questions.get(p["id"], {})
        extra = p["pred_set"] - p["gold_set"]
        print(f"\nID: {p['id']}")
        print(f"Event: {p['target_event'][:70]}...")
        print(f"Gold: {p['gold']} | Pred: {p['prediction']} | Extra: {extra}")
        for letter in extra:
            print(f"  {letter}: {q.get(f'option_{letter}', 'N/A')[:60]}...")

    # Save analysis
    analysis = {
        "summary": {
            "correct": len(correct),
            "partial": len(partial),
            "over_predict": len(over_predict),
            "wrong": len(wrong),
        },
        "recommendations": [
            "Multi-answer questions have ~50% error rate - need targeted prompt",
            "Single-answer questions have ~30% error rate - baseline works ok",
            "Model tends to under-predict more than over-predict",
            "Domain-specific routing may not help much (similar error rates)",
        ],
        "partial_errors": [{"id": p["id"], "gold": p["gold"], "pred": p["prediction"]} for p in partial],
        "over_predict_errors": [{"id": p["id"], "gold": p["gold"], "pred": p["prediction"]} for p in over_predict],
    }

    with open("experiments/error_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("\n\nAnalysis saved to experiments/error_analysis.json")


if __name__ == "__main__":
    main()
