#!/usr/bin/env python3
"""
Prepare RAG-enhanced training data.
Includes retrieved context in the training examples.
"""

import json
import re
from pathlib import Path

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
    """Find relevant sentences using keyword matching."""
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


def create_rag_training_example(question, context):
    """Create training example with RAG context."""
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

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": question['golden_answer']}
        ]
    }


def main():
    # Load training data
    with open("data/official/train_data/questions.jsonl") as f:
        train_questions = [json.loads(line) for line in f if line.strip()]

    with open("data/official/train_data/docs.json") as f:
        train_docs_list = json.load(f)
    train_docs = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(train_docs_list)}

    # Load dev data for validation
    with open("data/official/dev_data/questions.jsonl") as f:
        dev_questions = [json.loads(line) for line in f if line.strip()]

    with open("data/official/dev_data/docs.json") as f:
        dev_docs_list = json.load(f)
    dev_docs = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(dev_docs_list)}

    # Create RAG-enhanced training examples
    print(f"Processing {len(train_questions)} training examples...")
    train_examples = []
    context_count = 0
    for q in train_questions:
        topic_docs = train_docs.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)
        if context:
            context_count += 1
        example = create_rag_training_example(q, context)
        train_examples.append(example)

    print(f"Training examples with context: {context_count}/{len(train_questions)}")

    # Create validation examples
    print(f"Processing {len(dev_questions)} validation examples...")
    val_examples = []
    # Use first 182 dev examples for validation (same as before)
    for q in dev_questions[:182]:
        topic_docs = dev_docs.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)
        example = create_rag_training_example(q, context)
        val_examples.append(example)

    # Save training file
    with open("data/train_rag.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    # Save validation file
    with open("data/val_rag.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nCreated:")
    print(f"  data/train_rag.jsonl ({len(train_examples)} examples)")
    print(f"  data/val_rag.jsonl ({len(val_examples)} examples)")


if __name__ == "__main__":
    main()
