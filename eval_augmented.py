#!/usr/bin/env python3
"""Evaluate the augmented fine-tuned model on dev set."""

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

# Augmented model (trained on 3638 examples with shuffled options)
AUGMENTED_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:augmented:D2q6MoCj"

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
    return questions, docs_by_topic


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


def run_model(question, context, model=AUGMENTED_MODEL):
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
        model=model,
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default=AUGMENTED_MODEL)
    parser.add_argument("--output", default="experiments/augmented_dev_results.json")
    args = parser.parse_args()

    questions, docs_by_topic = load_data(args.split)

    print(f"Evaluating model on {len(questions)} {args.split} samples")
    print(f"Model: {args.model}\n")

    results = []
    scores = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)

        response = run_model(q, context, args.model)
        pred = parse_response(response)
        gold = set(q['golden_answer'].split(','))

        s = score_prediction(pred, gold)
        scores.append(s)

        results.append({
            "id": q["id"],
            "gold": list(gold),
            "pred": list(pred),
            "score": s
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"MODEL: {args.model}")
    print(f"{'='*60}")
    print(f"Average Score: {sum(scores)/len(scores):.4f}")
    print(f"Full Match: {sum(1 for s in scores if s == 1.0)}/{len(scores)} ({100*sum(1 for s in scores if s == 1.0)/len(scores):.1f}%)")
    print(f"Partial: {sum(1 for s in scores if s == 0.5)}/{len(scores)}")
    print(f"Wrong: {sum(1 for s in scores if s == 0.0)}/{len(scores)}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "avg_score": sum(scores)/len(scores),
            "full_match": sum(1 for s in scores if s == 1.0),
            "partial": sum(1 for s in scores if s == 0.5),
            "wrong": sum(1 for s in scores if s == 0.0),
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
