#!/usr/bin/env python3
"""
Post-processing rules based on error pattern analysis.

Patterns found:
1. Under-prediction of "A" in multi-answer cases (4/7 under-predict errors)
2. Capitol riot/protest events: misses "protests" as cause (3 errors)
3. Over-prediction on Wagner rebellion topics (2 errors)
4. Temporal confusion between similar events
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

FINETUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal::D2a6rPiD"

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

    # Load gold labels from questions (golden_answer field)
    gold = {}
    for q in questions:
        if "golden_answer" in q:
            gold[q["id"]] = set(q["golden_answer"].split(","))
    return questions, docs_by_topic, gold


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


def run_baseline(question, context):
    """Run the standard baseline model."""
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
        model=FINETUNED_MODEL,
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


# ==================== POST-PROCESSING RULES ====================

def is_protest_related(question):
    """Check if question is about protests/riots/demonstrations."""
    keywords = ['protest', 'riot', 'demonstration', 'storm', 'capitol', 'uprising', 'unrest']
    text = (question['target_event'] + ' ' +
            question['option_A'] + ' ' +
            question['option_B'] + ' ' +
            question['option_C'] + ' ' +
            question['option_D']).lower()
    return any(kw in text for kw in keywords)


def is_government_action(question):
    """Check if question is about government/official actions."""
    keywords = ['government', 'official', 'minister', 'president', 'fbi', 'atf',
                'congress', 'cleared', 'ordered', 'announced', 'declared']
    text = question['target_event'].lower()
    return any(kw in text for kw in keywords)


def option_mentions_protests(question, opt):
    """Check if an option mentions protests/demonstrations."""
    keywords = ['protest', 'demonstrat', 'gather', 'march', 'rally']
    text = question[f'option_{opt}'].lower()
    return any(kw in text for kw in keywords)


def rule_protest_cause(question, pred):
    """
    Rule 1: For protest-related events, if an option mentions protests and
    the prediction doesn't include it, consider adding it.
    """
    if not is_protest_related(question):
        return pred

    for opt in ['A', 'B', 'C', 'D']:
        if opt not in pred and option_mentions_protests(question, opt):
            # Check if this option happened BEFORE the event
            # (This is a heuristic - protest causes often start with verbs in past tense)
            option_text = question[f'option_{opt}']
            if any(w in option_text.lower() for w in ['erupted', 'protested', 'gathered', 'marched']):
                pred = pred | {opt}

    return pred


def rule_duplicate_options(question, pred):
    """
    Rule 2: If multiple options have identical or near-identical text,
    they should all be selected or none.
    """
    option_texts = {
        'A': question['option_A'].strip().lower(),
        'B': question['option_B'].strip().lower(),
        'C': question['option_C'].strip().lower(),
        'D': question['option_D'].strip().lower()
    }

    # Find duplicates
    for opt1 in ['A', 'B', 'C', 'D']:
        for opt2 in ['A', 'B', 'C', 'D']:
            if opt1 < opt2:
                # Check similarity (exact match or very similar)
                if option_texts[opt1] == option_texts[opt2]:
                    # If one is in pred, both should be
                    if opt1 in pred and opt2 not in pred:
                        pred = pred | {opt2}
                    elif opt2 in pred and opt1 not in pred:
                        pred = pred | {opt1}

    return pred


def rule_none_correct(question, pred):
    """
    Rule 3: "None of the others" option handling.
    If "None" option is selected, it should be alone.
    If other options are selected, "None" should not be.
    """
    none_option = None
    for opt in ['A', 'B', 'C', 'D']:
        if 'none of the others' in question[f'option_{opt}'].lower():
            none_option = opt
            break

    if none_option:
        if none_option in pred and len(pred) > 1:
            # Remove "None" if other options selected
            pred = pred - {none_option}
        elif pred == {none_option}:
            # Keep only "None" if it's the only selection
            pass

    return pred


def apply_all_rules(question, pred):
    """Apply all post-processing rules."""
    pred = rule_duplicate_options(question, pred)
    pred = rule_none_correct(question, pred)
    pred = rule_protest_cause(question, pred)
    return pred


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    args = parser.parse_args()

    questions, docs_by_topic, gold = load_data(args.split)

    if not gold:
        print("No gold labels available for this split")
        return

    if args.limit:
        questions = questions[:args.limit]

    print(f"Testing post-processing rules on {len(questions)} {args.split} samples")
    print(f"Model: {FINETUNED_MODEL}\n")

    baseline_scores = []
    postproc_scores = []
    changes = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)
        gold_answer = gold.get(q["id"], set())

        # Get baseline prediction
        baseline_response = run_baseline(q, context)
        baseline_pred = parse_response(baseline_response)

        # Apply post-processing
        postproc_pred = apply_all_rules(q, baseline_pred.copy())

        # Score both
        baseline_score = score_prediction(baseline_pred, gold_answer)
        postproc_score = score_prediction(postproc_pred, gold_answer)

        baseline_scores.append(baseline_score)
        postproc_scores.append(postproc_score)

        if baseline_pred != postproc_pred:
            changes.append({
                "id": q["id"],
                "event": q["target_event"][:80],
                "gold": list(gold_answer),
                "baseline": list(baseline_pred),
                "postproc": list(postproc_pred),
                "baseline_score": baseline_score,
                "postproc_score": postproc_score,
                "improved": postproc_score > baseline_score
            })

    # Print results
    print("\n" + "="*60)
    print(f"BASELINE:      {sum(baseline_scores)/len(baseline_scores):.4f}")
    print(f"  Full match: {sum(1 for s in baseline_scores if s == 1.0)}/{len(baseline_scores)}")

    print(f"\nPOST-PROCESS:  {sum(postproc_scores)/len(postproc_scores):.4f}")
    print(f"  Full match: {sum(1 for s in postproc_scores if s == 1.0)}/{len(postproc_scores)}")

    print(f"\nDIFFERENCE:    {(sum(postproc_scores)-sum(baseline_scores))/len(baseline_scores):+.4f}")

    print(f"\n{len(changes)} predictions changed:")
    improved = sum(1 for c in changes if c["improved"])
    worsened = sum(1 for c in changes if c["postproc_score"] < c["baseline_score"])
    same = len(changes) - improved - worsened
    print(f"  Improved: {improved}")
    print(f"  Worsened: {worsened}")
    print(f"  Same: {same}")

    if changes:
        print("\nChanged predictions:")
        for c in changes[:10]:
            status = "+" if c["improved"] else ("-" if c["postproc_score"] < c["baseline_score"] else "=")
            print(f"  [{status}] {c['id']}: {c['baseline']} -> {c['postproc']} (gold: {c['gold']})")

    # Save results
    output_path = f"experiments/postproc_{args.split}_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "baseline_avg": sum(baseline_scores)/len(baseline_scores),
            "postproc_avg": sum(postproc_scores)/len(postproc_scores),
            "changes": changes
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
