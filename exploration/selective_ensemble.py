#!/usr/bin/env python3
"""
Selective ensemble: Run multiple predictions at low temperature
and use majority voting only when there's disagreement.
"""

import json
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import sys

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

FINETUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal::D2a6rPiD"

CAUSAL_KEYWORDS = [
    'caused', 'because', 'due to', 'result of', 'led to', 'after',
    'following', 'triggered', 'sparked', 'prompted', 'resulted in',
    'consequence', 'therefore', 'thus', 'hence', 'as a result'
]


def load_data(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open(f"data/official/{split}_data/docs.json") as f:
        docs_list = json.load(f)
    docs_by_topic = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(docs_list)}

    gold = {}
    for q in questions:
        if "golden_answer" in q:
            gold[q["id"]] = set(q["golden_answer"].split(","))
    return questions, docs_by_topic, gold


def extract_keywords(text):
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'in', 'to',
                  'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through',
                  'and', 'or', 'but', 'if', 'then', 'that', 'this', 'these', 'those'}
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) > 3 and w not in stop_words]


def extract_relevant_context(question, docs, max_sentences=3):
    if not docs:
        return ""

    event_keywords = set(extract_keywords(question['target_event']))
    option_keywords = set()
    for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
        option_keywords.update(extract_keywords(question[opt]))
    all_keywords = event_keywords | option_keywords

    scored_sentences = []
    for doc in docs:
        text = doc.get("text", doc.get("content", doc.get("snippet", "")))
        for sent in re.split(r'[.!?]+', text):
            sent = sent.strip()
            if len(sent) < 30:
                continue

            sent_lower = sent.lower()
            score = 0
            sent_keywords = set(extract_keywords(sent))
            keyword_overlap = len(all_keywords & sent_keywords)
            score += keyword_overlap * 2

            for ck in CAUSAL_KEYWORDS:
                if ck in sent_lower:
                    score += 5
                    break

            if score > 0:
                scored_sentences.append((score, sent))

    scored_sentences.sort(reverse=True)
    top_sentences = [s for _, s in scored_sentences[:max_sentences]]
    return " ".join(top_sentences)


def run_model(question, context, temperature=0):
    """Run the model with specified temperature."""
    if context:
        prompt = f"""Context: {context}

What directly CAUSED this event? A cause must happen BEFORE the event.

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Answer with letter(s) only, comma-separated if multiple:"""
    else:
        prompt = f"""What directly CAUSED this event? A cause must happen BEFORE the event.

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Answer with letter(s) only, comma-separated if multiple:"""

    response = client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str) -> set[str]:
    response = response.upper().strip()
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    found = set(re.findall(r'\b([A-D])\b', response[-50:]))
    return found if found else {"A"}


def score_prediction(pred: set, gold: set) -> float:
    if pred == gold:
        return 1.0
    elif pred.issubset(gold) or gold.issubset(pred):
        return 0.5
    else:
        return 0.0


def majority_vote(predictions):
    """Take majority vote per option."""
    option_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for pred in predictions:
        for opt in pred:
            option_counts[opt] += 1

    threshold = len(predictions) / 2
    result = {opt for opt, count in option_counts.items() if count > threshold}

    # If nothing passes threshold, take the most common single prediction
    if not result:
        pred_strs = [",".join(sorted(p)) for p in predictions]
        most_common = Counter(pred_strs).most_common(1)[0][0]
        result = set(most_common.split(","))

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for ensemble")
    parser.add_argument("--temp", type=float, default=0.3, help="Temperature for varied runs")
    args = parser.parse_args()

    questions, docs_by_topic, gold = load_data(args.split)

    if not gold:
        print("No gold labels available for this split")
        return

    if args.limit:
        questions = questions[:args.limit]

    print(f"Testing selective ensemble on {len(questions)} {args.split} samples")
    print(f"Model: {FINETUNED_MODEL}")
    print(f"Runs: {args.runs}, Temperature: {args.temp}\n")

    baseline_scores = []
    ensemble_scores = []
    changed = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)
        gold_answer = gold.get(q["id"], set())

        # First, get baseline (temp=0)
        baseline_response = run_model(q, context, temperature=0)
        baseline_pred = parse_response(baseline_response)

        # Then, get multiple runs at low temp
        all_preds = [baseline_pred]
        for _ in range(args.runs - 1):
            response = run_model(q, context, temperature=args.temp)
            pred = parse_response(response)
            all_preds.append(pred)

        # Use majority vote
        ensemble_pred = majority_vote(all_preds)

        baseline_score = score_prediction(baseline_pred, gold_answer)
        ensemble_score = score_prediction(ensemble_pred, gold_answer)

        baseline_scores.append(baseline_score)
        ensemble_scores.append(ensemble_score)

        if baseline_pred != ensemble_pred:
            changed.append({
                "id": q["id"],
                "gold": list(gold_answer),
                "baseline": list(baseline_pred),
                "ensemble": list(ensemble_pred),
                "all_preds": [list(p) for p in all_preds],
                "baseline_score": baseline_score,
                "ensemble_score": ensemble_score
            })

    # Print results
    print("\n" + "="*60)
    print(f"BASELINE:  {sum(baseline_scores)/len(baseline_scores):.4f}")
    print(f"  Full match: {sum(1 for s in baseline_scores if s == 1.0)}/{len(baseline_scores)}")

    print(f"\nENSEMBLE:  {sum(ensemble_scores)/len(ensemble_scores):.4f}")
    print(f"  Full match: {sum(1 for s in ensemble_scores if s == 1.0)}/{len(ensemble_scores)}")

    print(f"\nDIFFERENCE: {(sum(ensemble_scores)-sum(baseline_scores))/len(baseline_scores):+.4f}")

    print(f"\n{len(changed)} predictions changed:")
    improved = sum(1 for c in changed if c["ensemble_score"] > c["baseline_score"])
    worsened = sum(1 for c in changed if c["ensemble_score"] < c["baseline_score"])
    same = len(changed) - improved - worsened
    print(f"  Improved: {improved}")
    print(f"  Worsened: {worsened}")
    print(f"  Same: {same}")

    if changed:
        print("\nChanged predictions (first 10):")
        for c in changed[:10]:
            status = "+" if c["ensemble_score"] > c["baseline_score"] else ("-" if c["ensemble_score"] < c["baseline_score"] else "=")
            print(f"  [{status}] {c['id']}: {c['baseline']} -> {c['ensemble']} (gold: {c['gold']})")
            print(f"      All preds: {c['all_preds']}")


if __name__ == "__main__":
    main()
