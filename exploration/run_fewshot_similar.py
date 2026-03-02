#!/usr/bin/env python3
"""
Few-shot with similar examples: Find similar training examples and use as demonstrations.
Based on research showing in-context learning helps LLMs.
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


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def question_to_text(q):
    return f"{q['target_event']} A:{q['option_A']} B:{q['option_B']} C:{q['option_C']} D:{q['option_D']}"


def build_example_bank(train_questions, max_examples=200):
    """Pre-compute embeddings for training examples."""
    print(f"Building example bank with {min(len(train_questions), max_examples)} examples...")
    bank = []
    for q in tqdm(train_questions[:max_examples], desc="Embedding"):
        text = question_to_text(q)
        emb = get_embedding(text)
        bank.append((q, emb))
    return bank


def find_similar_examples(query_q, example_bank, k=3):
    """Find k most similar training examples."""
    query_text = question_to_text(query_q)
    query_emb = get_embedding(query_text)

    scored = []
    for q, emb in example_bank:
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, q))

    scored.sort(reverse=True)
    return [q for _, q in scored[:k]]


def format_example(q):
    return f"""Event: {q['target_event']}
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}
Answer: {q['golden_answer']}"""


def run_fewshot(question, examples):
    examples_text = "\n\n".join([format_example(e) for e in examples])

    prompt = f"""Identify the direct cause(s) of each event. Multiple answers possible.

{examples_text}

Event: {question['target_event']}
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}
Answer:"""

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
    parser.add_argument("--k-examples", type=int, default=3)
    args = parser.parse_args()

    # Load training data for example bank
    train_questions = load_questions("train")
    example_bank = build_example_bank(train_questions, max_examples=200)

    # Load test questions
    test_questions = load_questions(args.split)[:args.max_samples]

    print(f"\nRunning FEW-SHOT (k={args.k_examples}) on {len(test_questions)} samples\n")

    scores_list = []
    results = []

    for q in tqdm(test_questions, desc="Evaluating"):
        similar = find_similar_examples(q, example_bank, k=args.k_examples)
        response = run_fewshot(q, similar)
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
    print(f"FEW-SHOT SIMILAR EXAMPLES RESULTS (k={args.k_examples})")
    print(f"{'='*60}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(test_questions)} ({full/len(test_questions)*100:.1f}%)")
    print(f"Baseline: 0.6825")
    print(f"{'='*60}")

    with open(f"experiments/fewshot_k{args.k_examples}_{args.split}_results.json", "w") as f:
        json.dump({"avg_score": avg, "k": args.k_examples, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
