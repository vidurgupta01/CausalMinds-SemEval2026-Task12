#!/usr/bin/env python3
"""
Run optimized experiments for SemEval 2026 Task 12.
"""

import argparse
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset, parse_answer, format_prediction
from src.baseline import score_prediction, evaluate_predictions
from src.optimized import (
    load_training_examples,
    optimized_few_shot_prompt,
    optimized_with_verification_prompt,
    self_consistency_prompt,
    parse_optimized_response,
    aggregate_votes,
)


def get_llm_response(prompt: str, model: str = "gpt-4o", temperature: float = 0.0) -> str:
    """Get response from an LLM API."""
    if model.startswith("gpt"):
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    elif model.startswith("claude"):
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unknown model: {model}")


def run_few_shot(
    split: str = "sample",
    model: str = "gpt-4o",
    n_examples: int = 5,
    max_samples: int = None,
    use_verification: bool = False,
):
    """Run few-shot optimized baseline."""
    print(f"Loading {split} dataset...")
    data = load_dataset(split)

    if max_samples:
        data = data[:max_samples]

    # Load training examples
    examples = load_training_examples(n_examples)
    print(f"Using {len(examples)} few-shot examples")

    prompt_fn = optimized_with_verification_prompt if use_verification else optimized_few_shot_prompt
    prompt_name = "verification" if use_verification else "few_shot"

    print(f"Running {prompt_name} with {model} on {len(data)} samples...")

    predictions = []
    golds = []
    results = []

    for item in tqdm(data):
        prompt = prompt_fn(item, examples)
        response = get_llm_response(prompt, model=model)
        pred = parse_optimized_response(response)
        gold = parse_answer(item["golden_answer"])

        predictions.append(pred)
        golds.append(gold)

        results.append({
            "id": item["id"],
            "target_event": item["target_event"],
            "prediction": format_prediction(pred),
            "gold": item["golden_answer"],
            "response": response,
        })

    metrics = evaluate_predictions(predictions, golds)
    print_results(f"{model} - {prompt_name} ({n_examples}-shot)", metrics)
    save_results(f"{model}_{prompt_name}_{n_examples}shot_{split}", metrics, results, {
        "model": model,
        "prompt_type": prompt_name,
        "n_examples": n_examples,
        "split": split,
        "num_samples": len(data),
    })

    return metrics


def run_self_consistency(
    split: str = "sample",
    model: str = "gpt-4o",
    n_samples: int = 5,
    max_samples: int = None,
    threshold: float = 0.5,
):
    """Run self-consistency (multiple samples + voting)."""
    print(f"Loading {split} dataset...")
    data = load_dataset(split)

    if max_samples:
        data = data[:max_samples]

    print(f"Running self-consistency ({n_samples} samples, threshold={threshold}) with {model}...")

    predictions = []
    golds = []
    results = []

    for item in tqdm(data):
        prompt = self_consistency_prompt(item)

        # Get multiple samples
        sample_preds = []
        sample_responses = []
        for _ in range(n_samples):
            response = get_llm_response(prompt, model=model, temperature=0.7)
            pred = parse_optimized_response(response)
            sample_preds.append(pred)
            sample_responses.append(response)

        # Aggregate votes
        final_pred = aggregate_votes(sample_preds, threshold)
        gold = parse_answer(item["golden_answer"])

        predictions.append(final_pred)
        golds.append(gold)

        results.append({
            "id": item["id"],
            "target_event": item["target_event"],
            "prediction": format_prediction(final_pred),
            "gold": item["golden_answer"],
            "sample_predictions": [format_prediction(p) for p in sample_preds],
            "responses": sample_responses,
        })

    metrics = evaluate_predictions(predictions, golds)
    print_results(f"{model} - self_consistency ({n_samples} samples)", metrics)
    save_results(f"{model}_self_consistency_{n_samples}samples_{split}", metrics, results, {
        "model": model,
        "prompt_type": "self_consistency",
        "n_samples": n_samples,
        "threshold": threshold,
        "split": split,
        "num_samples": len(data),
    })

    return metrics


def print_results(name: str, metrics: dict):
    """Print results in a nice format."""
    print("\n" + "=" * 60)
    print(f"Results: {name}")
    print("=" * 60)
    print(f"Average Score: {metrics['average_score']:.4f}")
    print(f"Full Match:    {metrics['full_match']}/{metrics['total']} ({metrics['full_match_rate']:.2%})")
    print(f"Partial Match: {metrics['partial_match']}/{metrics['total']} ({metrics['partial_match_rate']:.2%})")
    print(f"Incorrect:     {metrics['incorrect']}/{metrics['total']} ({metrics['incorrect_rate']:.2%})")
    print("=" * 60)


def save_results(name: str, metrics: dict, results: list, config: dict):
    """Save results to experiments directory."""
    output_path = Path("experiments")
    output_path.mkdir(exist_ok=True)

    with open(output_path / f"{name}_results.json", "w") as f:
        json.dump({
            "config": config,
            "metrics": metrics,
            "predictions": results,
        }, f, indent=2)

    print(f"Results saved to experiments/{name}_results.json")


def main():
    parser = argparse.ArgumentParser(description="Run optimized experiments")
    parser.add_argument("--method", default="few_shot",
                        choices=["few_shot", "verification", "self_consistency"])
    parser.add_argument("--split", default="sample", choices=["sample", "train", "dev", "test"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-examples", type=int, default=5, help="Few-shot examples")
    parser.add_argument("--n-samples", type=int, default=5, help="Self-consistency samples")
    parser.add_argument("--threshold", type=float, default=0.5, help="Voting threshold")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    if args.method == "few_shot":
        run_few_shot(
            split=args.split,
            model=args.model,
            n_examples=args.n_examples,
            max_samples=args.max_samples,
            use_verification=False,
        )
    elif args.method == "verification":
        run_few_shot(
            split=args.split,
            model=args.model,
            n_examples=args.n_examples,
            max_samples=args.max_samples,
            use_verification=True,
        )
    elif args.method == "self_consistency":
        run_self_consistency(
            split=args.split,
            model=args.model,
            n_samples=args.n_samples,
            max_samples=args.max_samples,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
