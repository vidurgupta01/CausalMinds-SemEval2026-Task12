"""
Data loading utilities for SemEval 2026 Task 12: Abductive Event Reasoning.
"""

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR


OFFICIAL_DATA_DIR = DATA_DIR / "official"


def load_questions(split: str = "train") -> list[dict[str, Any]]:
    """
    Load questions for a given split.

    Args:
        split: One of 'train', 'dev', 'test', 'sample'

    Returns:
        List of question instances, each containing:
        - id: Unique identifier (e.g., "q-101")
        - topic_id: Links to documents
        - target_event: The observed event
        - option_A through option_D: Candidate explanations
        - golden_answer: Correct option(s) like "A" or "A,B" (not in test)
    """
    filepath = OFFICIAL_DATA_DIR / f"{split}_data" / "questions.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    questions = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def load_docs(split: str = "train") -> dict[int, dict[str, Any]]:
    """
    Load documents for a given split.

    Args:
        split: One of 'train', 'dev', 'test', 'sample'

    Returns:
        Dictionary mapping topic_id to document data:
        - topic_id: Identifier
        - topic: Topic description
        - docs: List of retrieved documents
    """
    filepath = OFFICIAL_DATA_DIR / f"{split}_data" / "docs.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, "r") as f:
        docs_list = json.load(f)

    # Index by topic_id for easy lookup
    return {item["topic_id"]: item for item in docs_list}


def load_dataset(split: str = "train") -> list[dict[str, Any]]:
    """
    Load questions with their associated documents.

    Args:
        split: One of 'train', 'dev', 'test', 'sample'

    Returns:
        List of instances with questions and documents merged
    """
    questions = load_questions(split)
    docs = load_docs(split)

    for q in questions:
        topic_id = q["topic_id"]
        if topic_id in docs:
            q["documents"] = docs[topic_id]["docs"]
            q["topic"] = docs[topic_id]["topic"]
        else:
            q["documents"] = []
            q["topic"] = None

    return questions


def get_options(question: dict) -> list[str]:
    """Extract options A-D from a question."""
    return [
        question["option_A"],
        question["option_B"],
        question["option_C"],
        question["option_D"],
    ]


def parse_answer(answer_str: str) -> set[str]:
    """
    Parse answer string like "A,B" into set of labels.

    Args:
        answer_str: Comma-separated answer labels (e.g., "A", "A,B")

    Returns:
        Set of answer labels (e.g., {"A", "B"})
    """
    return set(label.strip() for label in answer_str.split(","))


def format_prediction(labels: set[str]) -> str:
    """
    Format prediction set as comma-separated string.

    Args:
        labels: Set of predicted labels (e.g., {"A", "B"})

    Returns:
        Sorted comma-separated string (e.g., "A,B")
    """
    return ",".join(sorted(labels))


def save_predictions(predictions: list[dict], output_path: Path) -> None:
    """
    Save predictions in submission format.

    Args:
        predictions: List of {"id": "q-X", "prediction": "A,B"} dicts
        output_path: Path to save predictions
    """
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"Predictions saved to {output_path}")
