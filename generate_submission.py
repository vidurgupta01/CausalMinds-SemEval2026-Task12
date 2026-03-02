#!/usr/bin/env python3
"""Generate submission file for CodaBench test set."""

import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import sys
import argparse
import shutil

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
    parser = argparse.ArgumentParser(description="Generate test submission")
    parser.add_argument("--model", required=True,
                       help=f"Model name or ID. Available: {', '.join(MODELS.keys())}")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG context")
    parser.add_argument("--output-dir", default="submissions", help="Output directory")
    args = parser.parse_args()

    # Resolve model
    model_id = MODELS.get(args.model, args.model)
    model_name = args.model if args.model in MODELS else "custom"

    # Load test data
    with open("data/official/test_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open("data/official/test_data/docs.json") as f:
        docs_list = json.load(f)
    docs_by_topic = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(docs_list)}

    print(f"Generating test predictions for {model_name}")
    print(f"Model ID: {model_id}")
    print(f"RAG: {'disabled' if args.no_rag else 'enabled'}")
    print(f"Questions: {len(questions)}\n")

    predictions = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = "" if args.no_rag else extract_relevant_context(q, topic_docs)

        response = run_model(q, context, model_id)
        pred = parse_response(response)

        predictions.append({
            "id": q["id"],
            "answer": ",".join(sorted(pred))
        })

    # Save predictions
    Path(args.output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file = f"{args.output_dir}/{model_name}_test_predictions.jsonl"

    with open(pred_file, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')
    print(f"\nPredictions saved to {pred_file}")

    # Create submission.jsonl
    submission_file = "submission.jsonl"
    with open(submission_file, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    # Create zip file
    Path("saved_submissions").mkdir(exist_ok=True)
    zip_name = f"saved_submissions/submission_{model_name}_test_{timestamp}"
    shutil.make_archive(zip_name, 'zip', '.', submission_file)
    print(f"Submission zip: {zip_name}.zip")

    # Show answer distribution
    from collections import Counter
    answers = [p["answer"] for p in predictions]
    print(f"\nAnswer distribution:")
    for ans, count in Counter(answers).most_common(10):
        print(f"  {ans}: {count}")


if __name__ == "__main__":
    main()
