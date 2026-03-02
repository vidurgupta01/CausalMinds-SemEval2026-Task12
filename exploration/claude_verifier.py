#!/usr/bin/env python3
"""
Use Claude as a verifier for GPT predictions.
Only ask Claude when there might be disagreement.
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
import anthropic

openai_client = OpenAI()
claude_client = anthropic.Anthropic()

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


def run_gpt(question, context):
    """Get prediction from fine-tuned GPT model."""
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

    response = openai_client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def run_claude_verifier(question, gpt_prediction, context, model="claude-opus-4-20250514"):
    """Ask Claude to verify GPT's prediction."""
    if context:
        prompt = f"""Context: {context}

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

A prediction system says the cause(s) are: {','.join(sorted(gpt_prediction))}

Your task: Verify if this is correct. A cause must happen BEFORE the event and directly lead to it.

Think step by step:
1. What is the event?
2. For each predicted cause, did it happen BEFORE the event?
3. Did it directly cause the event?
4. Are there any causes that were missed?

Then provide your final answer with letter(s) only, comma-separated if multiple:"""
    else:
        prompt = f"""Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

A prediction system says the cause(s) are: {','.join(sorted(gpt_prediction))}

Your task: Verify if this is correct. A cause must happen BEFORE the event and directly lead to it.

Think step by step:
1. What is the event?
2. For each predicted cause, did it happen BEFORE the event?
3. Did it directly cause the event?
4. Are there any causes that were missed?

Then provide your final answer with letter(s) only, comma-separated if multiple:"""

    response = claude_client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def run_openai_verifier(question, gpt_prediction, context, model="gpt-4.5-preview"):
    """Ask GPT-4.5/o3 to verify prediction."""
    if context:
        prompt = f"""Context: {context}

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

A prediction system says the cause(s) are: {','.join(sorted(gpt_prediction))}

Verify if this is correct. A cause must happen BEFORE the event and directly lead to it.
Answer with letter(s) only, comma-separated if multiple:"""
    else:
        prompt = f"""Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

A prediction system says the cause(s) are: {','.join(sorted(gpt_prediction))}

Verify if this is correct. A cause must happen BEFORE the event and directly lead to it.
Answer with letter(s) only, comma-separated if multiple:"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def parse_response(response: str) -> set[str]:
    response = response.upper().strip()
    # Look for final answer pattern
    if "FINAL ANSWER" in response.upper():
        response = response.split("FINAL ANSWER")[-1]
    # Get last line or last few characters
    lines = response.strip().split('\n')
    last_line = lines[-1] if lines else response

    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', last_line.strip())
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    found = set(re.findall(r'\b([A-D])\b', response[-100:]))
    return found if found else {"A"}


def score_prediction(pred: set, gold: set) -> float:
    if pred == gold:
        return 1.0
    elif pred.issubset(gold) or gold.issubset(pred):
        return 0.5
    else:
        return 0.0


def combine_predictions(gpt_pred, claude_pred, mode="conservative"):
    """Combine GPT and Claude predictions."""
    if mode == "conservative":
        # Only change if Claude strongly disagrees (completely different)
        if gpt_pred & claude_pred:  # If any overlap, trust GPT
            return gpt_pred
        else:  # No overlap, use Claude
            return claude_pred
    elif mode == "union":
        return gpt_pred | claude_pred
    elif mode == "intersection":
        result = gpt_pred & claude_pred
        return result if result else gpt_pred
    elif mode == "claude_override":
        return claude_pred
    return gpt_pred


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--mode", default="conservative",
                        choices=["conservative", "union", "intersection", "verifier_override"])
    parser.add_argument("--verifier", default="claude-opus",
                        choices=["claude-opus", "claude-sonnet", "gpt-4.5", "o3-mini", "gpt-4o"])
    args = parser.parse_args()

    verifier_models = {
        "claude-opus": "claude-opus-4-20250514",
        "claude-sonnet": "claude-sonnet-4-20250514",
        "gpt-4.5": "gpt-4.5-preview",
        "o3-mini": "o3-mini",
        "gpt-4o": "gpt-4o"
    }
    verifier_model = verifier_models[args.verifier]

    questions, docs_by_topic, gold = load_data(args.split)

    if not gold:
        print("No gold labels available for this split")
        return

    if args.limit:
        questions = questions[:args.limit]

    print(f"Testing verifier on {len(questions)} {args.split} samples")
    print(f"Verifier: {args.verifier} ({verifier_model})")
    print(f"Mode: {args.mode}\n")

    gpt_scores = []
    verifier_scores = []
    combined_scores = []
    details = []

    for q in tqdm(questions):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)
        gold_answer = gold.get(q["id"], set())

        # Get GPT prediction
        gpt_response = run_gpt(q, context)
        gpt_pred = parse_response(gpt_response)

        # Get verifier response
        if args.verifier.startswith("claude"):
            verifier_response = run_claude_verifier(q, gpt_pred, context, verifier_model)
        else:
            verifier_response = run_openai_verifier(q, gpt_pred, context, verifier_model)
        verifier_pred = parse_response(verifier_response)

        # Combine
        combined_pred = combine_predictions(gpt_pred, verifier_pred, args.mode)

        gpt_score = score_prediction(gpt_pred, gold_answer)
        verifier_score = score_prediction(verifier_pred, gold_answer)
        combined_score = score_prediction(combined_pred, gold_answer)

        gpt_scores.append(gpt_score)
        verifier_scores.append(verifier_score)
        combined_scores.append(combined_score)

        if gpt_pred != verifier_pred:
            details.append({
                "id": q["id"],
                "gold": list(gold_answer),
                "gpt": list(gpt_pred),
                "verifier": list(verifier_pred),
                "combined": list(combined_pred),
                "gpt_score": gpt_score,
                "verifier_score": verifier_score,
                "combined_score": combined_score
            })

    # Print results
    print("\n" + "="*60)
    print(f"GPT (fine-tuned): {sum(gpt_scores)/len(gpt_scores):.4f}")
    print(f"  Full match: {sum(1 for s in gpt_scores if s == 1.0)}/{len(gpt_scores)}")

    print(f"\nVerifier ({args.verifier}): {sum(verifier_scores)/len(verifier_scores):.4f}")
    print(f"  Full match: {sum(1 for s in verifier_scores if s == 1.0)}/{len(verifier_scores)}")

    print(f"\nCombined ({args.mode}): {sum(combined_scores)/len(combined_scores):.4f}")
    print(f"  Full match: {sum(1 for s in combined_scores if s == 1.0)}/{len(combined_scores)}")

    print(f"\n{len(details)} disagreements:")
    for d in details[:10]:
        print(f"  {d['id']}: GPT={d['gpt']} ({d['gpt_score']:.1f}), Ver={d['verifier']} ({d['verifier_score']:.1f}), Gold={d['gold']}")


if __name__ == "__main__":
    main()
