#!/usr/bin/env python3
"""
Smart router with targeted prompts based on error analysis:
1. Multi-answer questions: Use prompt that encourages finding ALL causes
2. "None" option questions: Use careful elimination prompt
3. Single-answer: Use standard prompt (works best)
4. High-overlap questions: Use disambiguation prompt
"""

import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from src.llm_engine import LLMEngine


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


# =============================================================================
# PROMPT VARIANTS
# =============================================================================

def standard_prompt(q):
    """Standard baseline prompt - works well for single-answer."""
    return f"""Identify the direct cause(s) of this event.

Event: {q['target_event']}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

A direct cause must:
1. Happen BEFORE the event
2. Have a clear causal mechanism that leads to the event

Answer with the letter(s) of the direct cause(s), comma-separated if multiple:"""


def multi_answer_prompt(q):
    """Prompt encouraging finding ALL causes - for questions likely to have multiple answers."""
    return f"""Identify ALL direct causes of this event. Multiple options can be correct.

Event: {q['target_event']}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

Consider EACH option independently:
- Did it happen BEFORE the event?
- Does it have a causal link to the event?
- Would the event still happen without it?

Include ALL options that are direct causes. Many events have multiple causes working together.

Answer (letters, comma-separated):"""


def none_option_prompt(q, none_letter):
    """Careful prompt for questions with "none of the others" option."""
    return f"""Carefully analyze whether any of these options caused the event.

Event: {q['target_event']}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

Note: Option {none_letter} says "None of the others are correct causes."

For each option A-D:
1. Did it happen BEFORE the event? (If after, it's an effect, not a cause)
2. Is there a direct causal mechanism?

If NO other option is a valid direct cause, select {none_letter}.
If one or more options ARE valid causes, select those (not {none_letter}).

Answer (letters only):"""


def disambiguation_prompt(q):
    """For questions with high word overlap - distinguish cause from effect."""
    return f"""IMPORTANT: Distinguish between CAUSES and EFFECTS.

Event: {q['target_event']}

Options:
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}

Remember:
- CAUSE = happened BEFORE the event and MADE it happen
- EFFECT = happened AFTER the event AS A RESULT of it
- Something related to the event is NOT necessarily its cause

Which option(s) are CAUSES (not effects)?

Answer (letters only):"""


# =============================================================================
# ROUTER LOGIC
# =============================================================================

def analyze_question(q):
    """Analyze question to determine routing."""
    event = q["target_event"].lower()
    options = [q[f"option_{l}"] for l in "ABCD"]
    options_lower = [o.lower() for o in options]

    # Check for "none of the others" option
    none_letter = None
    for i, o in enumerate(options_lower):
        if "none of the other" in o:
            none_letter = "ABCD"[i]
            break

    # Calculate word overlap between event and options
    event_words = set(re.findall(r'\b[a-z]{4,}\b', event))
    max_overlap = 0
    for o in options_lower:
        opt_words = set(re.findall(r'\b[a-z]{4,}\b', o))
        overlap = len(event_words & opt_words)
        max_overlap = max(max_overlap, overlap)

    # Count options with temporal indicators suggesting they're effects
    effect_indicators = ["after", "following", "resulted", "led to", "caused"]
    options_with_effects = sum(1 for o in options_lower if any(ind in o for ind in effect_indicators))

    return {
        "none_letter": none_letter,
        "max_overlap": max_overlap,
        "options_with_effects": options_with_effects,
    }


def route_question(q, analysis):
    """Decide which prompt to use based on analysis."""
    # Priority 1: Questions with "none" option need special handling
    if analysis["none_letter"]:
        return "none", none_option_prompt(q, analysis["none_letter"])

    # Priority 2: High overlap suggests disambiguation needed
    if analysis["max_overlap"] >= 3:
        return "disambiguation", disambiguation_prompt(q)

    # Priority 3: If multiple options look like potential causes, use multi-answer prompt
    # (We use multi-answer prompt as default since under-prediction is the main issue)
    return "multi_answer", multi_answer_prompt(q)


def parse_response(response: str) -> set[str]:
    """Parse response to extract letter answers."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="sample")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compare-baseline", action="store_true",
                       help="Also run baseline for comparison")
    args = parser.parse_args()

    questions = load_questions(args.split)
    if args.max_samples:
        questions = questions[:args.max_samples]

    engine = LLMEngine.from_model_name(args.model)

    print(f"Running SMART ROUTER on {len(questions)} samples")
    print(f"Model: {args.model}")
    print()

    scores_list = []
    baseline_scores = []
    results = []
    route_counts = {"none": 0, "disambiguation": 0, "multi_answer": 0, "standard": 0}

    for q in tqdm(questions):
        analysis = analyze_question(q)
        route_type, prompt = route_question(q, analysis)
        route_counts[route_type] += 1

        response = engine.get_response(prompt, max_tokens=64)
        pred = parse_response(response)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        result = {
            "id": q["id"],
            "route": route_type,
            "pred": ",".join(sorted(pred)),
            "gold": q["golden_answer"],
            "score": s,
        }

        # Optionally run baseline for comparison
        if args.compare_baseline:
            baseline_response = engine.get_response(standard_prompt(q), max_tokens=64)
            baseline_pred = parse_response(baseline_response)
            baseline_s = score(baseline_pred, gold)
            baseline_scores.append(baseline_s)
            result["baseline_pred"] = ",".join(sorted(baseline_pred))
            result["baseline_score"] = baseline_s

        results.append(result)

    # Print results
    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"SMART ROUTER RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(questions)}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{len(questions)} ({full/len(questions)*100:.1f}%)")
    print(f"Partial: {partial}/{len(questions)} ({partial/len(questions)*100:.1f}%)")
    print(f"Wrong: {wrong}/{len(questions)} ({wrong/len(questions)*100:.1f}%)")
    print(f"\nRoute distribution: {route_counts}")

    # Per-route accuracy
    print("\nAccuracy by route:")
    for route in route_counts:
        route_results = [r for r in results if r["route"] == route]
        if route_results:
            route_avg = sum(r["score"] for r in route_results) / len(route_results)
            print(f"  {route}: {route_avg:.4f} (n={len(route_results)})")

    if args.compare_baseline:
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        print(f"\nBaseline Average: {baseline_avg:.4f}")
        print(f"Router vs Baseline: {avg - baseline_avg:+.4f}")

        # Count wins/losses
        wins = sum(1 for r in results if r["score"] > r["baseline_score"])
        losses = sum(1 for r in results if r["score"] < r["baseline_score"])
        ties = sum(1 for r in results if r["score"] == r["baseline_score"])
        print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}")

    print(f"{'='*60}")

    # Save
    output = {
        "config": {
            "model": args.model,
            "method": "smart_router",
            "split": args.split,
            "num_samples": len(questions),
        },
        "metrics": {
            "average_score": avg,
            "full_match": full,
            "partial_match": partial,
            "incorrect": wrong,
            "route_counts": route_counts,
        },
        "predictions": results,
    }

    outfile = f"experiments/{args.model.replace('/', '-')}_smart_router_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
