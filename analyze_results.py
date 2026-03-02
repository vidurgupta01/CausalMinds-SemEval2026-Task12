#!/usr/bin/env python3
"""
Analyze experiment results and generate paper-ready artifacts.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict


EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")  # For paper artifacts


def load_experiment(filepath: Path) -> dict:
    """Load an experiment result file."""
    with open(filepath) as f:
        return json.load(f)


def analyze_errors(data: dict) -> dict:
    """Analyze error patterns in predictions."""
    predictions = data["predictions"]

    error_types = defaultdict(list)

    for pred in predictions:
        gold = set(pred["gold"].split(","))
        predicted = set(pred["prediction"].split(","))

        if predicted == gold:
            error_types["correct"].append(pred)
        elif predicted.issubset(gold):
            error_types["partial_correct"].append(pred)
        elif "C" in gold and "None" in pred.get("response", "").lower():
            # Missed "None of the above" cases
            error_types["missed_none"].append(pred)
        else:
            error_types["incorrect"].append(pred)

    return dict(error_types)


def generate_summary_table(experiments: list[dict]) -> str:
    """Generate a markdown summary table."""
    rows = []
    rows.append("| Model | Prompt | Docs | Split | N | Full Match | Partial | Incorrect | Avg Score |")
    rows.append("|-------|--------|------|-------|---|------------|---------|-----------|-----------|")

    for exp in experiments:
        config = exp["config"]
        metrics = exp["metrics"]

        row = (
            f"| {config['model']} "
            f"| {config['prompt_type']} "
            f"| {'Yes' if config['include_docs'] else 'No'} "
            f"| {config['split']} "
            f"| {config['num_samples']} "
            f"| {metrics['full_match_rate']:.1%} "
            f"| {metrics['partial_match_rate']:.1%} "
            f"| {metrics['incorrect_rate']:.1%} "
            f"| {metrics['average_score']:.4f} |"
        )
        rows.append(row)

    return "\n".join(rows)


def generate_latex_table(experiments: list[dict]) -> str:
    """Generate a LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline Results on SemEval 2026 Task 12}")
    lines.append(r"\label{tab:baseline_results}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Prompt & Docs & Split & Full & Partial & Incorrect & Score \\")
    lines.append(r"\midrule")

    for exp in experiments:
        config = exp["config"]
        metrics = exp["metrics"]

        line = (
            f"{config['model']} & "
            f"{config['prompt_type'].replace('_', ' ')} & "
            f"{'Yes' if config['include_docs'] else 'No'} & "
            f"{config['split']} & "
            f"{metrics['full_match_rate']:.1%} & "
            f"{metrics['partial_match_rate']:.1%} & "
            f"{metrics['incorrect_rate']:.1%} & "
            f"{metrics['average_score']:.3f} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_error_examples(data: dict, output_path: Path, n_examples: int = 10):
    """Save example errors for qualitative analysis."""
    errors = analyze_errors(data)

    examples = {
        "incorrect_examples": errors.get("incorrect", [])[:n_examples],
        "partial_examples": errors.get("partial_correct", [])[:n_examples],
        "correct_examples": errors.get("correct", [])[:n_examples],
    }

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)


def main():
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load all experiments
    experiments = []
    for filepath in EXPERIMENTS_DIR.glob("*_results.json"):
        exp = load_experiment(filepath)
        exp["filename"] = filepath.name
        experiments.append(exp)

    if not experiments:
        print("No experiment results found in experiments/")
        return

    # Sort by model name and split
    experiments.sort(key=lambda x: (x["config"]["model"], x["config"]["split"]))

    # Generate summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().isoformat()}")
    print(f"Total experiments: {len(experiments)}\n")

    # Markdown table
    md_table = generate_summary_table(experiments)
    print(md_table)

    # Save markdown summary
    summary_path = RESULTS_DIR / "results_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# SemEval 2026 Task 12 - Experiment Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Summary Table\n\n")
        f.write(md_table + "\n\n")
        f.write("## Experiment Details\n\n")
        for exp in experiments:
            f.write(f"### {exp['filename']}\n\n")
            f.write(f"- Model: {exp['config']['model']}\n")
            f.write(f"- Prompt: {exp['config']['prompt_type']}\n")
            f.write(f"- Include docs: {exp['config']['include_docs']}\n")
            f.write(f"- Split: {exp['config']['split']}\n")
            f.write(f"- Samples: {exp['config']['num_samples']}\n")
            f.write(f"- **Average Score: {exp['metrics']['average_score']:.4f}**\n\n")

    print(f"\nMarkdown summary saved to: {summary_path}")

    # Save LaTeX table
    latex_path = RESULTS_DIR / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write(generate_latex_table(experiments))
    print(f"LaTeX table saved to: {latex_path}")

    # Save error examples for each experiment
    for exp in experiments:
        error_path = RESULTS_DIR / f"{exp['filename'].replace('_results.json', '_errors.json')}"
        save_error_examples(exp, error_path)
        print(f"Error examples saved to: {error_path}")

    # Save combined JSON for all experiments
    combined_path = RESULTS_DIR / "all_experiments.json"
    with open(combined_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "experiments": [
                {
                    "filename": exp["filename"],
                    "config": exp["config"],
                    "metrics": exp["metrics"]
                }
                for exp in experiments
            ]
        }, f, indent=2)
    print(f"Combined results saved to: {combined_path}")

    print("\n" + "=" * 80)
    print("All artifacts saved to results/ directory")
    print("=" * 80)


if __name__ == "__main__":
    main()
