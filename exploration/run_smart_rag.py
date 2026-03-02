#!/usr/bin/env python3
"""
Smart RAG: Use embeddings to find only causally-relevant sentences from docs.
Based on CausalRAG paper insights.
"""

import json
import re
import numpy as np
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
    with open(f"data/official/{split}_data/docs.json") as f:
        docs_list = json.load(f)
    # Create topic_id -> docs mapping
    docs_by_topic = {d["topic_id"] if "topic_id" in d else i+1: d["docs"] for i, d in enumerate(docs_list)}
    return questions, docs_by_topic


def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding for text."""
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def extract_causal_context(question, docs, top_k=3):
    """Find most causally-relevant sentences from docs."""
    if not docs:
        return ""

    # Create query focused on causality
    query = f"What caused: {question['target_event']} Options: {question['option_A']}, {question['option_B']}, {question['option_C']}, {question['option_D']}"
    query_emb = get_embedding(query)

    # Extract sentences from all docs
    sentences = []
    for doc in docs:
        text = doc.get("text", doc.get("snippet", ""))
        # Split into sentences
        for sent in re.split(r'[.!?]+', text):
            sent = sent.strip()
            if len(sent) > 30:  # Skip very short sentences
                sentences.append(sent)

    if not sentences:
        return ""

    # Get embeddings and rank by similarity
    scored = []
    for sent in sentences[:50]:  # Limit to first 50 sentences for speed
        sent_emb = get_embedding(sent)
        score = cosine_similarity(query_emb, sent_emb)
        scored.append((score, sent))

    # Get top-k most relevant
    scored.sort(reverse=True)
    top_sentences = [s for _, s in scored[:top_k]]

    return " ".join(top_sentences)


def run_with_smart_rag(question, context):
    """Run inference with smart RAG context."""
    prompt = f"""Based on the context, identify what directly CAUSED this event.

Context: {context}

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

    print(f"Running SMART RAG on {len(questions)} samples")
    print("Using embeddings to find causally-relevant sentences\n")

    scores_list = []
    results = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_causal_context(q, topic_docs)

        response = run_with_smart_rag(q, context)
        pred = parse_response(response)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
            "context_used": context[:200] + "..." if context else "none"
        })

    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)

    print(f"\n{'='*60}")
    print(f"SMART RAG RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Baseline comparison: 0.6825 (dev)")
    print(f"{'='*60}")

    with open(f"experiments/smart_rag_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
