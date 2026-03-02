#!/usr/bin/env python3
"""Better RAG with improved retrieval for causal reasoning."""

import json
import re
import math
from pathlib import Path
from dotenv import load_dotenv
import sys
from collections import Counter

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

BEST_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:augmented:D2q6MoCj"

CAUSAL_PATTERNS = [
    r'\b(?:caused?|causing)\b',
    r'\bbecause\b',
    r'\bdue to\b',
    r'\b(?:result|resulted|resulting)\s+(?:of|in|from)\b',
    r'\bled to\b',
    r'\bafter\b',
    r'\bfollowing\b',
    r'\btriggered\b',
    r'\bsparked\b',
    r'\bprompted\b',
    r'\bconsequen(?:ce|tly)\b',
    r'\btherefore\b',
    r'\bthus\b',
    r'\bhence\b',
    r'\bas a result\b',
    r'\bin response to\b',
    r'\bstemm(?:ed|ing) from\b',
    r'\barising from\b',
    r'\bbrought about\b',
    r'\bgave rise to\b',
    r'\bowing to\b',
    r'\bon account of\b',
    r'\bby virtue of\b',
    r'\bled\s+\w+\s+to\b',
    r'\bforced\b',
    r'\bcompelled\b',
    r'\bnecessitated\b',
]


def tokenize(text):
    """Simple tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def compute_idf(docs_texts):
    """Compute IDF scores across all documents."""
    doc_count = len(docs_texts)
    df = Counter()
    for text in docs_texts:
        words = set(tokenize(text))
        for w in words:
            df[w] += 1

    idf = {}
    for word, count in df.items():
        idf[word] = math.log((doc_count + 1) / (count + 1)) + 1
    return idf


def bm25_score(query_tokens, doc_text, idf, k1=1.5, b=0.75, avg_dl=500):
    """BM25 scoring for a document against a query."""
    doc_tokens = tokenize(doc_text)
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)

    score = 0.0
    for qt in set(query_tokens):
        if qt in tf:
            term_idf = idf.get(qt, 1.0)
            term_tf = tf[qt]
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * dl / avg_dl)
            score += term_idf * numerator / denominator
    return score


def extract_better_context(question, docs, max_chunks=5):
    """Extract context using BM25 + causal pattern boosting."""
    if not docs:
        return ""

    # Build query from event + all options
    query_parts = [question['target_event']]
    for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
        query_parts.append(question[opt])
    query_text = " ".join(query_parts)
    query_tokens = tokenize(query_text)

    # Split documents into sentence chunks
    all_chunks = []
    all_doc_texts = []

    for doc in docs:
        text = doc.get("text", doc.get("content", doc.get("snippet", "")))
        all_doc_texts.append(text)

        # Split into sentences, keeping some context
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Create overlapping chunks of 2-3 sentences for better context
        for i in range(len(sentences)):
            chunk = sentences[i]
            if len(chunk.strip()) < 20:
                continue
            # Add next sentence if available for context
            if i + 1 < len(sentences):
                chunk = sentences[i] + " " + sentences[i+1]
            all_chunks.append(chunk)

    if not all_chunks:
        return ""

    # Compute IDF from all documents
    idf = compute_idf(all_doc_texts)

    # Compute average document length
    avg_dl = sum(len(tokenize(c)) for c in all_chunks) / len(all_chunks) if all_chunks else 100

    # Score each chunk
    scored_chunks = []
    for chunk in all_chunks:
        # BM25 relevance score
        relevance = bm25_score(query_tokens, chunk, idf, avg_dl=avg_dl)

        # Causal pattern bonus
        causal_bonus = 0
        chunk_lower = chunk.lower()
        for pattern in CAUSAL_PATTERNS:
            if re.search(pattern, chunk_lower):
                causal_bonus += 3

        # Event mention bonus (if the event or key entities mentioned)
        event_tokens = set(tokenize(question['target_event']))
        chunk_tokens = set(tokenize(chunk))
        event_overlap = len(event_tokens & chunk_tokens)
        event_bonus = event_overlap * 1.5

        total_score = relevance + causal_bonus + event_bonus

        if total_score > 0:
            scored_chunks.append((total_score, chunk))

    scored_chunks.sort(reverse=True)

    # Deduplicate: skip chunks that overlap too much with already selected ones
    selected = []
    selected_tokens = set()

    for score, chunk in scored_chunks:
        chunk_toks = set(tokenize(chunk))
        # Skip if >70% overlap with already selected content
        if selected_tokens and len(chunk_toks & selected_tokens) / max(len(chunk_toks), 1) > 0.7:
            continue
        selected.append(chunk)
        selected_tokens.update(chunk_toks)
        if len(selected) >= max_chunks:
            break

    return " ".join(selected)


def parse_response(response: str) -> set:
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


def run_model(question, context, model=BEST_MODEL):
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default=BEST_MODEL)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-chunks", type=int, default=5)
    args = parser.parse_args()

    with open(f"data/official/{args.split}_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open(f"data/official/{args.split}_data/docs.json") as f:
        docs_list = json.load(f)
    docs_by_topic = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(docs_list)}

    print(f"Better RAG evaluation on {len(questions)} {args.split} samples")
    print(f"Model: {args.model}")
    print(f"Max context chunks: {args.max_chunks}\n")

    results = []
    scores = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_better_context(q, topic_docs, max_chunks=args.max_chunks)

        response = run_model(q, context, args.model)
        pred = parse_response(response)

        has_gold = "golden_answer" in q
        if has_gold:
            gold = set(q['golden_answer'].split(','))
            s = score_prediction(pred, gold)
            scores.append(s)

        result = {
            "id": q["id"],
            "pred": list(pred),
        }
        if has_gold:
            result["gold"] = list(gold)
            result["score"] = s

        results.append(result)

    if scores:
        print(f"\n{'='*60}")
        print(f"BETTER RAG RESULTS")
        print(f"{'='*60}")
        print(f"Average Score: {sum(scores)/len(scores):.4f}")
        print(f"Full Match: {sum(1 for s in scores if s == 1.0)}/{len(scores)} ({100*sum(1 for s in scores if s == 1.0)/len(scores):.1f}%)")
        print(f"Partial: {sum(1 for s in scores if s == 0.5)}/{len(scores)}")
        print(f"Wrong: {sum(1 for s in scores if s == 0.0)}/{len(scores)}")

    output_file = args.output or f"experiments/better_rag_{args.split}_results.json"
    Path("experiments").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "max_chunks": args.max_chunks,
            "avg_score": sum(scores)/len(scores) if scores else None,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
