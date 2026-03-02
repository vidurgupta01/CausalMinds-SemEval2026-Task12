#!/usr/bin/env python3
"""
Keyword-based RAG: Faster than embedding-based, uses keyword matching.
Looks for causal keywords and event-related terms.
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

# Causal keywords
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
    """Extract important keywords from text."""
    # Remove common words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'in', 'to',
                  'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through',
                  'and', 'or', 'but', 'if', 'then', 'that', 'this', 'these', 'those'}
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) > 3 and w not in stop_words]


def extract_relevant_context(question, docs, max_sentences=5):
    """Find relevant sentences using keyword matching."""
    if not docs:
        return ""

    # Get keywords from event and options
    event_keywords = set(extract_keywords(question['target_event']))
    option_keywords = set()
    for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
        option_keywords.update(extract_keywords(question[opt]))

    all_keywords = event_keywords | option_keywords

    # Score sentences
    scored_sentences = []
    for doc in docs:
        text = doc.get("text", doc.get("snippet", ""))
        for sent in re.split(r'[.!?]+', text):
            sent = sent.strip()
            if len(sent) < 30:
                continue

            sent_lower = sent.lower()
            score = 0

            # Keyword matches
            sent_keywords = set(extract_keywords(sent))
            keyword_overlap = len(all_keywords & sent_keywords)
            score += keyword_overlap * 2

            # Causal keyword bonus
            for ck in CAUSAL_KEYWORDS:
                if ck in sent_lower:
                    score += 5
                    break

            if score > 0:
                scored_sentences.append((score, sent))

    # Sort and get top sentences
    scored_sentences.sort(reverse=True)
    top_sentences = [s for _, s in scored_sentences[:max_sentences]]
    return " ".join(top_sentences)


def run_with_context(question, context):
    """Run inference with context."""
    prompt = f"""Based on the context, identify what directly CAUSED this event.

Context: {context if context else "No additional context."}

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

A direct cause must happen BEFORE the event and lead to it.
Answer with letter(s) only, comma-separated if multiple:"""

    response = client.chat.completions.create(
        model="gpt-4o",
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
    args = parser.parse_args()

    questions, docs_by_topic = load_data(args.split)
    questions = questions[:args.max_samples]

    print(f"Running KEYWORD RAG on {len(questions)} samples\n")

    scores_list = []
    results = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)

        response = run_with_context(q, context)
        pred = parse_response(response)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
        })

    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)

    print(f"\n{'='*60}")
    print(f"KEYWORD RAG RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Baseline (sample): 0.5000")
    print(f"{'='*60}")

    with open(f"experiments/keyword_rag_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
