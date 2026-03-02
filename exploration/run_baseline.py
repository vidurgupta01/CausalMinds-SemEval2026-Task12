#!/usr/bin/env python3
"""
Run baseline experiments for SemEval 2026 Task 12.
"""

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset, parse_answer, format_prediction
from src.baseline import (
    zero_shot_prompt,
    chain_of_thought_prompt,
    parse_response,
    evaluate_predictions,
)
from src.llm_engine import LLMEngine


def run_baseline(
    split: str = "sample",
    model: str = "gpt-4",
    prompt_type: str = "zero_shot",
    include_docs: bool = False,
    max_samples: int = None,
    output_dir: str = "experiments",
):
    """
    Run baseline evaluation.

    Args:
        split: Dataset split to evaluate on
        model: LLM model to use
        prompt_type: "zero_shot" or "chain_of_thought"
        include_docs: Whether to include retrieved documents
        max_samples: Maximum number of samples to evaluate (None for all)
        output_dir: Directory to save results
    """
    print(f"Loading {split} dataset...")
    data = load_dataset(split)

    if max_samples:
        data = data[:max_samples]

    print(f"Running {prompt_type} baseline with {model} on {len(data)} samples...")

    # Initialize LLM engine
    engine = LLMEngine.from_model_name(model)
    prompt_fn = zero_shot_prompt if prompt_type == "zero_shot" else chain_of_thought_prompt

    predictions = []
    golds = []
    results = []

    for item in tqdm(data):
        prompt = prompt_fn(item, include_docs=include_docs)
        response = engine.get_response(prompt, max_tokens=512)
        pred = parse_response(response)
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

    # Evaluate
    metrics = evaluate_predictions(predictions, golds)

    print("\n" + "=" * 50)
    print(f"Results for {model} - {prompt_type}")
    print("=" * 50)
    print(f"Average Score: {metrics['average_score']:.4f}")
    print(f"Full Match:    {metrics['full_match']}/{metrics['total']} ({metrics['full_match_rate']:.2%})")
    print(f"Partial Match: {metrics['partial_match']}/{metrics['total']} ({metrics['partial_match_rate']:.2%})")
    print(f"Incorrect:     {metrics['incorrect']}/{metrics['total']} ({metrics['incorrect_rate']:.2%})")
    print("=" * 50)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_name = f"{model}_{prompt_type}_{'with_docs' if include_docs else 'no_docs'}_{split}"

    with open(output_path / f"{experiment_name}_results.json", "w") as f:
        json.dump({
            "config": {
                "model": model,
                "prompt_type": prompt_type,
                "include_docs": include_docs,
                "split": split,
                "num_samples": len(data),
            },
            "metrics": metrics,
            "predictions": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path / f'{experiment_name}_results.json'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--split", default="sample", choices=["sample", "train", "dev", "test"])
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--prompt", default="zero_shot", choices=["zero_shot", "chain_of_thought"])
    parser.add_argument("--include-docs", action="store_true", help="Include retrieved documents")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output-dir", default="experiments", help="Output directory")

    args = parser.parse_args()

    run_baseline(
        split=args.split,
        model=args.model,
        prompt_type=args.prompt,
        include_docs=args.include_docs,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
