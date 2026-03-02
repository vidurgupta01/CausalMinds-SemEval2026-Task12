#!/usr/bin/env python3
"""
Advanced multi-stage approach combining:
1. Smart RAG - extract relevant context from docs
2. Multi-model ensemble (GPT-4o + Claude)
3. Chain-of-thought reasoning with verification
4. Confidence-weighted voting
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import sys

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from src.llm_engine import LLMEngine


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_docs(split):
    """Load documents indexed by topic_id."""
    try:
        with open(f"data/official/{split}_data/docs.json") as f:
            docs_list = json.load(f)
        return {d["topic_id"]: d["docs"] for d in docs_list}
    except:
        return {}


def extract_relevant_context(question, docs_by_topic, max_chars=1500):
    """
    Smart RAG: Extract only relevant sentences from docs.
    Focus on sentences that mention the event or options.
    """
    topic_id = question.get("topic_id")
    if not topic_id or topic_id not in docs_by_topic:
        return ""

    docs = docs_by_topic[topic_id]
    event_lower = question["target_event"].lower()

    # Key terms from event and options
    key_terms = set()
    for text in [question["target_event"]] + [question[f"option_{l}"] for l in "ABCD"]:
        # Extract significant words (length > 4, not common words)
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        key_terms.update(words)

    # Remove common words
    common = {"which", "would", "could", "should", "their", "there", "about", "after", "before", "other", "these", "those"}
    key_terms -= common

    relevant_sentences = []

    for doc in docs[:3]:  # Only first 3 docs
        content = doc.get("content", "")
        sentences = re.split(r'(?<=[.!?])\s+', content)

        for sent in sentences:
            sent_lower = sent.lower()
            # Score sentence by key term matches
            score = sum(1 for term in key_terms if term in sent_lower)
            if score >= 2:  # At least 2 key terms
                relevant_sentences.append((score, sent.strip()))

    # Sort by relevance and take top sentences
    relevant_sentences.sort(reverse=True, key=lambda x: x[0])

    context = ""
    for _, sent in relevant_sentences[:5]:
        if len(context) + len(sent) < max_chars:
            context += sent + " "

    return context.strip()


# =============================================================================
# STAGE 1: Temporal Analysis
# =============================================================================
def stage1_temporal_prompt(q):
    return f"""Analyze the TEMPORAL ORDER of these events.

TARGET EVENT: {q['target_event']}

For each option below, determine if it happened BEFORE or AFTER the target event:

A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

For each option, respond with:
- BEFORE: if this option happened before the target event (potential cause)
- AFTER: if this option happened after the target event (effect, NOT a cause)
- UNCLEAR: if temporal order cannot be determined

Format your response as:
A: [BEFORE/AFTER/UNCLEAR] - [brief reason]
B: [BEFORE/AFTER/UNCLEAR] - [brief reason]
C: [BEFORE/AFTER/UNCLEAR] - [brief reason]
D: [BEFORE/AFTER/UNCLEAR] - [brief reason]"""


# =============================================================================
# STAGE 2: Causal Mechanism Analysis
# =============================================================================
def stage2_causal_prompt(q, temporal_analysis):
    return f"""Given the temporal analysis, now analyze CAUSAL MECHANISMS.

TARGET EVENT: {q['target_event']}

Previous temporal analysis:
{temporal_analysis}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

For each option that was marked BEFORE or UNCLEAR, evaluate:
1. Is there a clear causal mechanism connecting this to the target event?
2. Would the target event have happened WITHOUT this option?

Format:
A: [CAUSE/NOT_CAUSE] - [mechanism explanation]
B: [CAUSE/NOT_CAUSE] - [mechanism explanation]
C: [CAUSE/NOT_CAUSE] - [mechanism explanation]
D: [CAUSE/NOT_CAUSE] - [mechanism explanation]"""


# =============================================================================
# STAGE 3: Final Decision with Context
# =============================================================================
def stage3_final_prompt(q, temporal_analysis, causal_analysis, context=""):
    context_section = f"\nRelevant background information:\n{context}\n" if context else ""

    return f"""Make the FINAL DECISION on direct causes.

TARGET EVENT: {q['target_event']}
{context_section}
Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

Previous analysis:
TEMPORAL: {temporal_analysis}
CAUSAL: {causal_analysis}

Based on all analysis, which option(s) are DIRECT CAUSES of the target event?

Rules:
- A direct cause MUST happen BEFORE the event
- A direct cause MUST have a clear causal mechanism
- Multiple causes are possible
- If an option says "None of the others" and no other option is a valid cause, select it

FINAL ANSWER (letters only, comma-separated if multiple):"""


# =============================================================================
# Verification Stage
# =============================================================================
def verification_prompt(q, initial_answer):
    return f"""VERIFY this answer for logical consistency.

TARGET EVENT: {q['target_event']}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

PROPOSED ANSWER: {initial_answer}

Check each selected option:
1. Does it happen BEFORE the target event? (If after, it's an EFFECT, not a cause)
2. Is there a clear causal link?
3. Did we miss any other valid causes?

If the answer is correct, respond: VERIFIED: {initial_answer}
If corrections needed, respond: CORRECTED: [new answer letters]"""


def parse_response(response: str) -> set[str]:
    """Parse response to extract letter answers."""
    response = response.upper().strip()

    # Look for VERIFIED: or CORRECTED: patterns
    match = re.search(r'(?:VERIFIED|CORRECTED):\s*([A-D](?:\s*,\s*[A-D])*)', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))

    # Look for FINAL ANSWER pattern
    match = re.search(r'FINAL\s*ANSWER[:\s]*([A-D](?:\s*,\s*[A-D])*)', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))

    # Try exact match at end
    match = re.search(r'([A-D](?:\s*,\s*[A-D])*)\s*$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))

    # Find any letters in last 100 chars
    found = set(re.findall(r'\b([A-D])\b', response[-100:]))
    return found if found else {"A"}


def score(pred, gold):
    if pred == gold:
        return 1.0
    elif pred and pred.issubset(gold):
        return 0.5
    return 0.0


def run_multi_stage(q, engine, docs_by_topic, use_context=True):
    """Run multi-stage reasoning pipeline."""
    annotations = {}

    # Extract relevant context
    context = ""
    if use_context:
        context = extract_relevant_context(q, docs_by_topic)
        annotations["context_extracted"] = context[:200] + "..." if len(context) > 200 else context

    # Stage 1: Temporal analysis
    temporal_prompt = stage1_temporal_prompt(q)
    temporal_response = engine.get_response(temporal_prompt, max_tokens=300)
    annotations["stage1_temporal"] = temporal_response

    # Stage 2: Causal mechanism analysis
    causal_prompt = stage2_causal_prompt(q, temporal_response)
    causal_response = engine.get_response(causal_prompt, max_tokens=300)
    annotations["stage2_causal"] = causal_response

    # Stage 3: Final decision
    final_prompt = stage3_final_prompt(q, temporal_response, causal_response, context)
    final_response = engine.get_response(final_prompt, max_tokens=100)
    annotations["stage3_final"] = final_response
    initial_pred = parse_response(final_response)

    # Verification stage
    verify_prompt = verification_prompt(q, ",".join(sorted(initial_pred)))
    verify_response = engine.get_response(verify_prompt, max_tokens=150)
    annotations["stage4_verification"] = verify_response
    final_pred = parse_response(verify_response)

    annotations["initial_prediction"] = ",".join(sorted(initial_pred))
    annotations["final_prediction"] = ",".join(sorted(final_pred))

    return final_pred, annotations


def run_multi_model_ensemble(q, engines, docs_by_topic, use_context=True):
    """Run with multiple models and ensemble their predictions."""
    all_predictions = []
    all_annotations = {}

    for name, engine in engines.items():
        pred, annotations = run_multi_stage(q, engine, docs_by_topic, use_context)
        all_predictions.append(pred)
        all_annotations[name] = annotations

    # Weighted voting (each model gets equal weight)
    vote_counts = Counter()
    for pred in all_predictions:
        for letter in pred:
            vote_counts[letter] += 1

    # Include letters that appear in majority of models
    threshold = len(engines) / 2
    final_pred = {letter for letter, count in vote_counts.items() if count >= threshold}

    if not final_pred and vote_counts:
        final_pred = {vote_counts.most_common(1)[0][0]}

    all_annotations["ensemble_votes"] = dict(vote_counts)
    all_annotations["individual_preds"] = [",".join(sorted(p)) for p in all_predictions]

    return final_pred if final_pred else {"A"}, all_annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="sample")
    parser.add_argument("--models", default="gpt-4o", help="Comma-separated model names")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-context", action="store_true", help="Disable RAG context")
    parser.add_argument("--single-stage", action="store_true", help="Skip multi-stage, use simple prompt")
    args = parser.parse_args()

    questions = load_questions(args.split)
    docs_by_topic = load_docs(args.split)

    if args.max_samples:
        questions = questions[:args.max_samples]

    # Initialize engines
    model_names = [m.strip() for m in args.models.split(",")]
    engines = {name: LLMEngine.from_model_name(name) for name in model_names}

    print(f"Running ADVANCED MULTI-STAGE approach on {len(questions)} samples")
    print(f"Models: {model_names}")
    print(f"RAG Context: {'Disabled' if args.no_context else 'Enabled'}")
    print(f"Multi-stage: {'Disabled' if args.single_stage else 'Enabled'}")
    print()

    scores_list = []
    results = []

    for q in tqdm(questions):
        if len(engines) == 1:
            pred, annotations = run_multi_stage(
                q, list(engines.values())[0], docs_by_topic,
                use_context=not args.no_context
            )
        else:
            pred, annotations = run_multi_model_ensemble(
                q, engines, docs_by_topic,
                use_context=not args.no_context
            )

        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "event": q["target_event"][:80],
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
            "annotations": annotations,
        })

    # Print results
    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"ADVANCED MULTI-STAGE RESULTS")
    print(f"{'='*60}")
    print(f"Models: {model_names}")
    print(f"Samples: {len(questions)}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"{'='*60}")

    # Save with full annotations
    model_tag = "_".join(m.split("/")[-1].replace("-", "") for m in model_names)
    output = {
        "config": {
            "models": model_names,
            "method": "advanced_multi_stage",
            "use_context": not args.no_context,
            "split": args.split,
            "num_samples": len(questions),
        },
        "metrics": {
            "average_score": avg,
            "full_match": full,
            "partial_match": partial,
            "incorrect": wrong,
            "total": len(questions),
        },
        "predictions": results,
    }

    outfile = f"experiments/advanced_{model_tag}_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")
    print("Full annotations included for each prediction.")


if __name__ == "__main__":
    main()
