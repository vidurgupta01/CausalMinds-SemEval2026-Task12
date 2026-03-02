#!/usr/bin/env python3
"""
Combined: Smart RAG + Structured JSON Output.
Best of both approaches.
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
    docs_by_topic = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(docs_list)}
    return questions, docs_by_topic


def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def extract_causal_context(question, docs, top_k=5):
    """Find most causally-relevant sentences from docs."""
    if not docs:
        return ""

    query = f"What caused: {question['target_event']}"
    query_emb = get_embedding(query)

    sentences = []
    for doc in docs:
        text = doc.get("text", doc.get("snippet", ""))
        for sent in re.split(r'[.!?]+', text):
            sent = sent.strip()
            if len(sent) > 30:
                sentences.append(sent)

    if not sentences:
        return ""

    scored = []
    for sent in sentences[:50]:
        sent_emb = get_embedding(sent)
        score = cosine_similarity(query_emb, sent_emb)
        scored.append((score, sent))

    scored.sort(reverse=True)
    top_sentences = [s for _, s in scored[:top_k]]
    return " ".join(top_sentences)


def run_with_smart_rag_structured(question, context):
    """Run with smart RAG context and structured JSON output."""
    prompt = f"""Based on the context, determine which options are direct CAUSES of this event.
A direct cause must: (1) happen BEFORE the event, (2) directly lead to the event.

Context: {context if context else "No additional context available."}

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Return a JSON object with:
- "reasoning": Brief explanation for your choices
- "causes": Array of letters (A/B/C/D) that are direct causes"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str) -> set[str]:
    try:
        data = json.loads(response)
        causes = data.get("causes", [])
        if causes:
            return set(c.upper() for c in causes if c.upper() in ['A', 'B', 'C', 'D'])
    except:
        pass
    found = set(re.findall(r'\b([A-D])\b', response.upper()[-100:]))
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

    print(f"Running SMART RAG + STRUCTURED on {len(questions)} samples\n")

    scores_list = []
    results = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_causal_context(q, topic_docs)

        response = run_with_smart_rag_structured(q, context)
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
    print(f"SMART RAG + STRUCTURED RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Baseline: 0.6825")
    print(f"{'='*60}")

    with open(f"experiments/smart_rag_structured_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
