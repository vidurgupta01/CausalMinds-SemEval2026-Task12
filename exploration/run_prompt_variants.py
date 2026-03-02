#!/usr/bin/env python3
"""
Test small prompt variations against baseline.
Goal: Find slight improvements without over-engineering.
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
# PROMPT VARIANTS - Small tweaks to baseline
# =============================================================================

PROMPTS = {
    "baseline": """Identify the direct cause(s) of this event.

Event: {event}

Options:
A. {A}
B. {B}
C. {C}
D. {D}

A direct cause must:
1. Happen BEFORE the event
2. Have a clear causal mechanism that leads to the event

Answer with the letter(s) of the direct cause(s), comma-separated if multiple:""",

    "v1_shorter": """What directly CAUSED this event? (Multiple answers possible)

Event: {event}

A. {A}
B. {B}
C. {C}
D. {D}

Answer (letters only):""",

    "v2_temporal": """Identify what CAUSED this event. A cause must happen BEFORE the event.

Event: {event}

A. {A}
B. {B}
C. {C}
D. {D}

Which happened BEFORE and CAUSED the event? (letters only, comma-separated if multiple):""",

    "v3_explicit_multi": """Identify ALL direct causes of this event. There may be multiple.

Event: {event}

A. {A}
B. {B}
C. {C}
D. {D}

Direct cause = happened BEFORE + made the event happen.
Answer with ALL correct letters:""",

    "v4_elimination": """Event: {event}

A. {A}
B. {B}
C. {C}
D. {D}

For each option: Did it happen BEFORE the event AND cause it?
Select the direct cause(s):""",
}


def format_prompt(template, q):
    return template.format(
        event=q["target_event"],
        A=q["option_A"],
        B=q["option_B"],
        C=q["option_C"],
        D=q["option_D"],
    )


def parse_response(response: str) -> set[str]:
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
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    questions = load_questions(args.split)[:args.max_samples]
    engine = LLMEngine.from_model_name(args.model)

    print(f"Testing {len(PROMPTS)} prompt variants on {len(questions)} samples")
    print(f"Model: {args.model}\n")

    all_results = {}

    for name, template in PROMPTS.items():
        print(f"Running {name}...")
        scores_list = []

        for q in tqdm(questions, desc=name):
            prompt = format_prompt(template, q)
            response = engine.get_response(prompt, max_tokens=64)
            pred = parse_response(response)
            gold = set(q["golden_answer"].split(","))
            scores_list.append(score(pred, gold))

        avg = sum(scores_list) / len(scores_list)
        full = sum(1 for s in scores_list if s == 1.0)

        all_results[name] = {
            "avg_score": avg,
            "full_match": full,
            "full_match_pct": full / len(questions) * 100,
        }
        print(f"  -> {avg:.4f} ({full}/{len(questions)} full match)\n")

    # Summary
    print("=" * 60)
    print("SUMMARY - Prompt Variant Comparison")
    print("=" * 60)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["avg_score"], reverse=True)
    for name, metrics in sorted_results:
        print(f"{name:20s}: {metrics['avg_score']:.4f} ({metrics['full_match_pct']:.1f}% full)")

    best = sorted_results[0]
    baseline = all_results["baseline"]
    print(f"\nBest: {best[0]} ({best[1]['avg_score']:.4f})")
    print(f"vs Baseline: {best[1]['avg_score'] - baseline['avg_score']:+.4f}")

    # Save
    with open(f"experiments/prompt_variants_{args.split}_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to experiments/prompt_variants_{args.split}_results.json")


if __name__ == "__main__":
    main()
