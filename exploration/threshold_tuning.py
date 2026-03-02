#!/usr/bin/env python3
"""
Threshold tuning for fine-tuned model.
Get per-option confidence and test different thresholds.
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

    # Load gold labels from questions (golden_answer field)
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


def get_option_confidence(question, context, option_letter):
    """Get confidence score for a specific option being a cause."""
    option_text = question[f'option_{option_letter}']

    if context:
        prompt = f"""Context: {context}

Is this a direct CAUSE of the event? A cause must happen BEFORE the event.

Event: {question['target_event']}

Potential cause: {option_letter}. {option_text}

Answer YES if this directly caused the event, NO if it did not."""
    else:
        prompt = f"""Is this a direct CAUSE of the event? A cause must happen BEFORE the event.

Event: {question['target_event']}

Potential cause: {option_letter}. {option_text}

Answer YES if this directly caused the event, NO if it did not."""

    response = client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
        logprobs=True,
        top_logprobs=5
    )

    content = response.choices[0].message.content.strip().upper()

    # Get logprobs for YES/NO
    logprobs = response.choices[0].logprobs
    if logprobs and logprobs.content:
        first_token_logprobs = logprobs.content[0].top_logprobs
        yes_prob = 0
        no_prob = 0
        for lp in first_token_logprobs:
            token = lp.token.upper().strip()
            prob = 2.718281828 ** lp.logprob  # exp(logprob)
            if token.startswith('YES') or token == 'Y':
                yes_prob = max(yes_prob, prob)
            elif token.startswith('NO') or token == 'N':
                no_prob = max(no_prob, prob)

        # Return confidence as probability of YES
        if yes_prob + no_prob > 0:
            return yes_prob / (yes_prob + no_prob)

    # Fallback: binary based on response
    return 1.0 if content.startswith('YES') else 0.0


def get_all_options_confidence(question, context):
    """Get confidence for all 4 options."""
    confidences = {}
    for opt in ['A', 'B', 'C', 'D']:
        confidences[opt] = get_option_confidence(question, context, opt)
    return confidences


def run_baseline(question, context):
    """Run the standard baseline model to get predictions."""
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
        temperature=0
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


def apply_threshold(confidences, threshold):
    """Select options above threshold."""
    selected = {opt for opt, conf in confidences.items() if conf >= threshold}
    # Always select at least one (highest confidence)
    if not selected:
        selected = {max(confidences, key=confidences.get)}
    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--mode", choices=["confidence", "baseline", "both"], default="both")
    args = parser.parse_args()

    questions, docs_by_topic, gold = load_data(args.split)

    if not gold:
        print("No gold labels available for this split")
        return

    questions = questions[:args.limit]
    print(f"Testing threshold tuning on {len(questions)} {args.split} samples")
    print(f"Model: {FINETUNED_MODEL}\n")

    results = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)
        gold_answer = gold.get(q["id"], set())

        result = {
            "id": q["id"],
            "gold": gold_answer
        }

        if args.mode in ["baseline", "both"]:
            baseline_response = run_baseline(q, context)
            result["baseline_pred"] = parse_response(baseline_response)

        if args.mode in ["confidence", "both"]:
            confidences = get_all_options_confidence(q, context)
            result["confidences"] = confidences

        results.append(result)

    # Analyze results
    print("\n" + "="*60)

    if args.mode in ["baseline", "both"]:
        baseline_scores = [score_prediction(r["baseline_pred"], r["gold"]) for r in results]
        print(f"BASELINE: {sum(baseline_scores)/len(baseline_scores):.4f}")
        print(f"  Full match: {sum(1 for s in baseline_scores if s == 1.0)}/{len(baseline_scores)}")

    if args.mode in ["confidence", "both"]:
        print("\nTHRESHOLD TUNING:")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        for thresh in thresholds:
            scores = []
            for r in results:
                pred = apply_threshold(r["confidences"], thresh)
                scores.append(score_prediction(pred, r["gold"]))
            avg = sum(scores) / len(scores)
            full = sum(1 for s in scores if s == 1.0)
            print(f"  Threshold {thresh}: {avg:.4f} ({full}/{len(scores)} full match)")

        # Find optimal threshold
        best_thresh = 0.5
        best_score = 0
        for thresh in [i/20 for i in range(1, 20)]:  # 0.05 to 0.95
            scores = []
            for r in results:
                pred = apply_threshold(r["confidences"], thresh)
                scores.append(score_prediction(pred, r["gold"]))
            avg = sum(scores) / len(scores)
            if avg > best_score:
                best_score = avg
                best_thresh = thresh

        print(f"\n  OPTIMAL: threshold={best_thresh:.2f}, score={best_score:.4f}")

    # Save detailed results
    output_path = f"experiments/threshold_tuning_{args.split}_{args.limit}_results.json"
    with open(output_path, "w") as f:
        # Convert sets to lists for JSON
        for r in results:
            r["gold"] = list(r["gold"])
            if "baseline_pred" in r:
                r["baseline_pred"] = list(r["baseline_pred"])
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
