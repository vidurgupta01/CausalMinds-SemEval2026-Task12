#!/usr/bin/env python3
"""Evaluate any fine-tuned model on dev or test set."""

import json
import re
from pathlib import Path
from dotenv import load_dotenv
import sys
import argparse

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

# Available models
MODELS = {
    "original_ft": "ft:gpt-4.1-mini-2025-04-14:personal::D2UIiELC",
    "augmented_ft": "ft:gpt-4.1-mini-2025-04-14:personal:augmented:D2q6MoCj",
    "combined3ep_ft": "ft:gpt-4.1-mini-2025-04-14:personal:combined3ep:D320DHl2",
    "5epoch_ft": "ft:gpt-4.1-mini-2025-04-14:personal:5epochs:D2kZfdPv",
    "1epoch_ft": "ft:gpt-4.1-mini-2025-04-14:personal:1epoch:D39xO9cd",
    "gpt4o": "gpt-4o",
}

CAUSAL_KEYWORDS = [
    'caused', 'because', 'due to', 'result of', 'led to', 'after',
    'following', 'triggered', 'sparked', 'prompted', 'resulted in',
    'consequence', 'therefore', 'thus', 'hence', 'as a result'
]


def extract_keywords(text):
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'in', 'to',
                  'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through',
                  'and', 'or', 'but', 'if', 'then', 'that', 'this', 'these', 'those'}
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) > 3 and w not in stop_words]


def extract_relevant_context(question, docs, max_sentences=3):
    """Extract relevant context using keyword matching."""
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


def parse_response(response: str) -> set:
    """Parse model response to extract answer letters."""
    response = response.upper().strip()
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    found = set(re.findall(r'\b([A-D])\b', response[-50:]))
    return found if found else {"A"}


def score_prediction(pred: set, gold: set) -> float:
    """Score a prediction against gold standard."""
    if pred == gold:
        return 1.0
    elif pred.issubset(gold) or gold.issubset(pred):
        return 0.5
    else:
        return 0.0


def run_model(question, context, model_id):
    """Run model on a single question."""
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
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on dev/test set")
    parser.add_argument("--model", required=True,
                       help=f"Model name or ID. Available: {', '.join(MODELS.keys())}")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG context")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    # Resolve model
    model_id = MODELS.get(args.model, args.model)
    model_name = args.model if args.model in MODELS else "custom"

    # Load data
    with open(f"data/official/{args.split}_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open(f"data/official/{args.split}_data/docs.json") as f:
        docs_list = json.load(f)
    docs_by_topic = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(docs_list)}

    print(f"Evaluating {model_name} on {len(questions)} {args.split} samples")
    print(f"Model ID: {model_id}")
    print(f"RAG: {'disabled' if args.no_rag else 'enabled'}\n")

    results = []
    scores = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = "" if args.no_rag else extract_relevant_context(q, topic_docs)

        response = run_model(q, context, model_id)
        pred = parse_response(response)

        has_gold = "golden_answer" in q
        if has_gold:
            gold = set(q['golden_answer'].split(','))
            s = score_prediction(pred, gold)
            scores.append(s)

        result = {
            "id": q["id"],
            "pred": ",".join(sorted(pred)),
        }
        if has_gold:
            result["gold"] = q['golden_answer']
            result["score"] = s

        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*60}")

    if scores:
        print(f"Average Score: {sum(scores)/len(scores):.4f}")
        print(f"Full Match: {sum(1 for s in scores if s == 1.0)}/{len(scores)} ({100*sum(1 for s in scores if s == 1.0)/len(scores):.1f}%)")
        print(f"Partial: {sum(1 for s in scores if s == 0.5)}/{len(scores)}")
        print(f"Wrong: {sum(1 for s in scores if s == 0.0)}/{len(scores)}")

        # Show errors
        print(f"\nErrors:")
        for r in results:
            if "score" in r and r["score"] < 1.0:
                print(f"  {r['id']}: pred={r['pred']}, gold={r['gold']}, score={r['score']}")

    # Save results
    output_file = args.output or f"experiments/{model_name}_{args.split}_results.json"
    Path("experiments").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "model": model_id,
            "model_name": model_name,
            "split": args.split,
            "rag_enabled": not args.no_rag,
            "avg_score": sum(scores)/len(scores) if scores else None,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
