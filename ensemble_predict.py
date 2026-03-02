#!/usr/bin/env python3
"""Ensemble predictions from multiple models using majority voting."""

import json
import re
from pathlib import Path
from dotenv import load_dotenv
import sys
from collections import Counter

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

# All available fine-tuned models
MODELS = {
    "original_ft": "ft:gpt-4.1-mini-2025-04-14:personal::D2UIiELC",
    "augmented_ft": "ft:gpt-4.1-mini-2025-04-14:personal:augmented:D2q6MoCj",
    "combined3ep_ft": "ft:gpt-4.1-mini-2025-04-14:personal:combined3ep:D320DHl2",
    "5epoch_ft": "ft:gpt-4.1-mini-2025-04-14:personal:5epochs:D2kZfdPv",
    "gpt4o": "gpt-4o",
}


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


def parse_response(response: str) -> set:
    response = response.upper().strip()
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    found = set(re.findall(r'\b([A-D])\b', response[-50:]))
    return found if found else {"A"}


def run_model(question, context, model_name, model_id):
    """Run a single model on a question."""
    if model_name == "gpt4o":
        # Use a more detailed prompt for base models
        if context:
            prompt = f"""You are an expert in causal reasoning about news events.

Context from news articles: {context}

Given the following event, identify which option(s) directly CAUSED the event.
A cause must happen BEFORE the event and directly lead to it.
Be careful: effects/consequences happen AFTER the event and are NOT causes.

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Think step by step about temporal order. Answer with ONLY the letter(s), comma-separated if multiple:"""
        else:
            prompt = f"""You are an expert in causal reasoning about news events.

Given the following event, identify which option(s) directly CAUSED the event.
A cause must happen BEFORE the event and directly lead to it.
Be careful: effects/consequences happen AFTER the event and are NOT causes.

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Think step by step about temporal order. Answer with ONLY the letter(s), comma-separated if multiple:"""
    else:
        # Fine-tuned model prompt (matches training format)
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


def ensemble_vote(predictions_per_model):
    """Combine predictions using weighted majority voting.

    Strategy: For each option (A,B,C,D), count how many models include it.
    Include it in final answer if majority of models agree.
    """
    option_votes = Counter()
    n_models = len(predictions_per_model)

    for pred_set in predictions_per_model:
        for letter in pred_set:
            option_votes[letter] += 1

    # Include options that majority agrees on
    threshold = n_models / 2.0
    result = set()
    for letter in ['A', 'B', 'C', 'D']:
        if option_votes.get(letter, 0) > threshold:
            result.add(letter)

    # Fallback: if no option passes threshold, use the most common full prediction
    if not result:
        full_preds = [','.join(sorted(p)) for p in predictions_per_model]
        most_common = Counter(full_preds).most_common(1)[0][0]
        result = set(most_common.split(','))

    return result


def score_prediction(pred: set, gold: set) -> float:
    if pred == gold:
        return 1.0
    elif pred.issubset(gold) or gold.issubset(pred):
        return 0.5
    else:
        return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--models", nargs="+",
                       default=["original_ft", "augmented_ft", "combined3ep_ft"],
                       help="Models to ensemble")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load data
    with open(f"data/official/{args.split}_data/questions.jsonl") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open(f"data/official/{args.split}_data/docs.json") as f:
        docs_list = json.load(f)
    docs_by_topic = {d.get("topic_id", i+1): d["docs"] for i, d in enumerate(docs_list)}

    selected_models = {name: MODELS[name] for name in args.models if name in MODELS}
    print(f"Ensemble of {len(selected_models)} models on {len(questions)} {args.split} samples:")
    for name, model_id in selected_models.items():
        print(f"  - {name}: {model_id}")
    print()

    # Get predictions from each model
    all_model_preds = {name: [] for name in selected_models}

    for q in tqdm(questions, desc="Running ensemble"):
        topic_docs = docs_by_topic.get(q.get("topic_id"), [])
        context = extract_relevant_context(q, topic_docs)

        for name, model_id in selected_models.items():
            response = run_model(q, context, name, model_id)
            pred = parse_response(response)
            all_model_preds[name].append(pred)

    # Compute ensemble predictions
    results = []
    scores = []

    for i, q in enumerate(questions):
        model_preds = [all_model_preds[name][i] for name in selected_models]
        ensemble_pred = ensemble_vote(model_preds)

        has_gold = "golden_answer" in q
        if has_gold:
            gold = set(q['golden_answer'].split(','))
            s = score_prediction(ensemble_pred, gold)
            scores.append(s)

        result = {
            "id": q["id"],
            "ensemble_pred": ",".join(sorted(ensemble_pred)),
        }

        # Add individual model predictions
        for name in selected_models:
            result[f"pred_{name}"] = ",".join(sorted(all_model_preds[name][i]))

        if has_gold:
            result["gold"] = q['golden_answer']
            result["score"] = s

            # Show disagreements
            individual_scores = {}
            for name in selected_models:
                individual_scores[name] = score_prediction(all_model_preds[name][i], gold)
            result["individual_scores"] = individual_scores

        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS ({', '.join(selected_models.keys())})")
    print(f"{'='*60}")

    if scores:
        print(f"Ensemble Score: {sum(scores)/len(scores):.4f}")
        print(f"Full Match: {sum(1 for s in scores if s == 1.0)}/{len(scores)} ({100*sum(1 for s in scores if s == 1.0)/len(scores):.1f}%)")
        print(f"Partial: {sum(1 for s in scores if s == 0.5)}/{len(scores)}")
        print(f"Wrong: {sum(1 for s in scores if s == 0.0)}/{len(scores)}")

        # Individual model scores
        print(f"\nIndividual model scores:")
        for name in selected_models:
            model_scores = []
            for i, q in enumerate(questions):
                if "golden_answer" in q:
                    gold = set(q['golden_answer'].split(','))
                    model_scores.append(score_prediction(all_model_preds[name][i], gold))
            if model_scores:
                print(f"  {name}: {sum(model_scores)/len(model_scores):.4f}")

    # Save results
    output_file = args.output or f"experiments/ensemble_{'_'.join(selected_models.keys())}_{args.split}_results.json"
    Path("experiments").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "models": list(selected_models.keys()),
            "split": args.split,
            "avg_score": sum(scores)/len(scores) if scores else None,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
