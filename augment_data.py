#!/usr/bin/env python3
"""
Data augmentation for training data.
Uses GPT-4o to paraphrase events and options while preserving causal relationships.
"""

import json
import re
import random
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

CAUSAL_KEYWORDS = [
    'caused', 'because', 'due to', 'result of', 'led to', 'after',
    'following', 'triggered', 'sparked', 'prompted', 'resulted in',
    'consequence', 'therefore', 'thus', 'hence', 'as a result'
]


def load_training_data():
    """Load original training data."""
    with open("data/official/train_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return questions


def extract_keywords(text):
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'in', 'to',
                  'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through',
                  'and', 'or', 'but', 'if', 'then', 'that', 'this', 'these', 'those'}
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) > 3 and w not in stop_words]


def load_docs():
    """Load document context."""
    with open("data/official/train_data/docs.json") as f:
        docs_list = json.load(f)
    return {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(docs_list)}


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


def paraphrase_question(question, context=""):
    """Use GPT-4o to paraphrase the event and options."""
    prompt = f"""Paraphrase this causal reasoning question. Keep the same meaning but use different wording.
Preserve all proper nouns, dates, and numbers exactly. Keep the causal relationships intact.

Original event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Correct answer(s): {question['golden_answer']}

Provide paraphrased versions in this exact JSON format:
{{
    "event": "paraphrased event",
    "option_A": "paraphrased option A",
    "option_B": "paraphrased option B",
    "option_C": "paraphrased option C",
    "option_D": "paraphrased option D"
}}

Important: The answer must remain the same ({question['golden_answer']}). Only change wording, not meaning."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error paraphrasing: {e}")
        return None


def create_hard_negative(question):
    """Create a version with shuffled options to make it harder."""
    # Shuffle the option letters but keep the answer mapping correct
    options = ['A', 'B', 'C', 'D']
    shuffled = options.copy()
    random.shuffle(shuffled)

    mapping = dict(zip(options, shuffled))
    reverse_mapping = {v: k for k, v in mapping.items()}

    new_q = question.copy()
    new_q['option_A'] = question[f'option_{reverse_mapping["A"]}']
    new_q['option_B'] = question[f'option_{reverse_mapping["B"]}']
    new_q['option_C'] = question[f'option_{reverse_mapping["C"]}']
    new_q['option_D'] = question[f'option_{reverse_mapping["D"]}']

    # Update the answer
    old_answers = set(question['golden_answer'].split(','))
    new_answers = {mapping[a] for a in old_answers}
    new_q['golden_answer'] = ','.join(sorted(new_answers))

    return new_q


def format_for_finetuning(question, context=""):
    """Format question for fine-tuning."""
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Number of examples to augment")
    parser.add_argument("--output", default="data/train_augmented.jsonl")
    parser.add_argument("--paraphrase", action="store_true", help="Use GPT-4o paraphrasing (slow, costly)")
    parser.add_argument("--shuffle", action="store_true", help="Add shuffled option versions")
    args = parser.parse_args()

    print("Loading training data...")
    questions = load_training_data()
    docs_by_topic = load_docs()

    print(f"Original training examples: {len(questions)}")

    augmented = []

    # First, add all original examples with RAG context
    print("\nAdding original examples with RAG context...")
    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)
        augmented.append(format_for_finetuning(q, context))

    # Add shuffled versions
    if args.shuffle:
        print("\nAdding shuffled option versions...")
        for q in tqdm(questions):
            shuffled_q = create_hard_negative(q)
            topic_docs = docs_by_topic.get(q.get("topic_id"), [])
            context = extract_relevant_context(shuffled_q, topic_docs)
            augmented.append(format_for_finetuning(shuffled_q, context))

    # Add paraphrased versions (expensive)
    if args.paraphrase:
        print(f"\nParaphrasing {min(args.limit, len(questions))} examples...")
        to_paraphrase = questions[:args.limit]

        for q in tqdm(to_paraphrase):
            topic_docs = docs_by_topic.get(q.get("topic_id"), [])
            context = extract_relevant_context(q, topic_docs)

            paraphrased = paraphrase_question(q, context)
            if paraphrased:
                new_q = q.copy()
                new_q['target_event'] = paraphrased.get('event', q['target_event'])
                new_q['option_A'] = paraphrased.get('option_A', q['option_A'])
                new_q['option_B'] = paraphrased.get('option_B', q['option_B'])
                new_q['option_C'] = paraphrased.get('option_C', q['option_C'])
                new_q['option_D'] = paraphrased.get('option_D', q['option_D'])

                # Recalculate context for paraphrased version
                new_context = extract_relevant_context(new_q, topic_docs)
                augmented.append(format_for_finetuning(new_q, new_context))

    print(f"\nTotal augmented examples: {len(augmented)}")

    # Shuffle and save
    random.shuffle(augmented)

    with open(args.output, 'w') as f:
        for item in augmented:
            f.write(json.dumps(item) + '\n')

    print(f"Saved to {args.output}")

    # Create validation set (10% of original)
    val_size = len(questions) // 10
    val_questions = questions[-val_size:]

    val_output = args.output.replace('.jsonl', '_val.jsonl')
    with open(val_output, 'w') as f:
        for q in val_questions:
            topic_docs = docs_by_topic.get(q.get("topic_id"), [])
            context = extract_relevant_context(q, topic_docs)
            item = format_for_finetuning(q, context)
            f.write(json.dumps(item) + '\n')

    print(f"Validation set saved to {val_output} ({val_size} examples)")


if __name__ == "__main__":
    main()
