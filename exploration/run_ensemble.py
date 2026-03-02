#!/usr/bin/env python3
"""
Run ensemble and persona-based experiments.
"""

import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble import (
    PERSONAS,
    detect_domain,
    make_persona_prompt,
    make_ensemble_prompts,
    ensemble_vote,
    route_strategy,
    ENSEMBLE_PROMPTS,
)
from src.llm_engine import LLMEngine


def load_questions(split):
    with open(f"data/official/{split}_data/questions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_response(response: str) -> set[str]:
    """Parse response to extract letter answers."""
    response = response.upper().strip()
    # Try exact match first
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))
    # Find any letters
    found = set(re.findall(r'\b([A-D])\b', response[-50:]))
    return found if found else {"A"}


def score(pred, gold):
    if pred == gold:
        return 1.0
    elif pred and pred.issubset(gold):
        return 0.5
    return 0.0


def run_persona(questions, engine):
    """Run persona-based approach with domain routing."""
    print("Running PERSONA approach (domain-routed)...")

    scores_list = []
    results = []
    domain_counts = {}

    for q in tqdm(questions):
        domain = detect_domain(q)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        system_msg, user_msg = make_persona_prompt(q, domain)
        response = engine.get_response(user_msg, system=system_msg)
        pred = parse_response(response)
        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "domain": domain,
            "gold": q["golden_answer"],
            "pred": ",".join(sorted(pred)),
            "score": s,
        })

    print(f"\nDomain distribution: {domain_counts}")
    return scores_list, results


def run_ensemble(questions, engine):
    """Run ensemble approach with multiple prompts and voting."""
    print("Running ENSEMBLE approach (4 prompts + voting)...")

    scores_list = []
    results = []

    for q in tqdm(questions):
        prompts = make_ensemble_prompts(q)
        predictions = []

        for prompt in prompts:
            response = engine.get_response(prompt)
            pred = parse_response(response)
            predictions.append(pred)

        # Vote
        final_pred = ensemble_vote(predictions)
        gold = set(q["golden_answer"].split(","))
        s = score(final_pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "individual_preds": [",".join(sorted(p)) for p in predictions],
            "final_pred": ",".join(sorted(final_pred)),
            "gold": q["golden_answer"],
            "score": s,
        })

    return scores_list, results


def run_routed(questions, engine):
    """Run routed approach - picks best strategy per question."""
    print("Running ROUTED approach (auto-select strategy)...")

    scores_list = []
    results = []
    strategy_counts = {}

    for q in tqdm(questions):
        strategy = route_strategy(q)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if strategy == "ensemble":
            # Use ensemble
            prompts = make_ensemble_prompts(q)
            predictions = []
            for prompt in prompts:
                response = engine.get_response(prompt)
                predictions.append(parse_response(response))
            pred = ensemble_vote(predictions)

        elif strategy == "elimination":
            # Use elimination prompt
            prompt = ENSEMBLE_PROMPTS["elimination"].format(
                event=q["target_event"],
                A=q["option_A"], B=q["option_B"],
                C=q["option_C"], D=q["option_D"],
            )
            response = engine.get_response(prompt)
            pred = parse_response(response)

        else:  # persona:domain
            domain = strategy.split(":")[1]
            system_msg, user_msg = make_persona_prompt(q, domain)
            response = engine.get_response(user_msg, system=system_msg)
            pred = parse_response(response)

        gold = set(q["golden_answer"].split(","))
        s = score(pred, gold)
        scores_list.append(s)

        results.append({
            "id": q["id"],
            "strategy": strategy,
            "gold": q["golden_answer"],
            "pred": ",".join(sorted(pred)),
            "score": s,
        })

    print(f"\nStrategy distribution: {strategy_counts}")
    return scores_list, results


def print_results(name, scores_list, n):
    avg = sum(scores_list) / len(scores_list)
    full = sum(1 for s in scores_list if s == 1.0)
    partial = sum(1 for s in scores_list if s == 0.5)
    wrong = sum(1 for s in scores_list if s == 0.0)

    print(f"\n{'='*60}")
    print(f"{name} RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {n}")
    print(f"Average Score: {avg:.4f}")
    print(f"Full Match: {full}/{n} ({full/n*100:.1f}%)")
    print(f"Partial: {partial}/{n} ({partial/n*100:.1f}%)")
    print(f"Wrong: {wrong}/{n} ({wrong/n*100:.1f}%)")
    print(f"{'='*60}")

    return {"avg": avg, "full": full, "partial": partial, "wrong": wrong}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="persona",
                       choices=["persona", "ensemble", "routed", "all"])
    parser.add_argument("--split", default="sample")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    questions = load_questions(args.split)
    if args.max_samples:
        questions = questions[:args.max_samples]

    print(f"Testing on {len(questions)} samples from {args.split}")
    print(f"Model: {args.model}")
    print()

    # Initialize LLM engine
    engine = LLMEngine.from_model_name(args.model)
    all_results = {}

    if args.method in ["persona", "all"]:
        scores, results = run_persona(questions, engine)
        metrics = print_results("PERSONA", scores, len(questions))
        all_results["persona"] = {"metrics": metrics, "results": results}

    if args.method in ["ensemble", "all"]:
        scores, results = run_ensemble(questions, engine)
        metrics = print_results("ENSEMBLE", scores, len(questions))
        all_results["ensemble"] = {"metrics": metrics, "results": results}

    if args.method in ["routed", "all"]:
        scores, results = run_routed(questions, engine)
        metrics = print_results("ROUTED", scores, len(questions))
        all_results["routed"] = {"metrics": metrics, "results": results}

    # Save results
    outfile = f"experiments/{args.model}_{args.method}_{args.split}_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
