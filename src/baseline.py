"""
Baseline approaches for SemEval 2026 Task 12: Abductive Event Reasoning.
"""

from typing import Any


OPTION_LABELS = ["A", "B", "C", "D"]


def zero_shot_prompt(question: dict, include_docs: bool = False) -> str:
    """
    Generate a zero-shot prompt for cause identification.

    Args:
        question: Question dict with target_event, options, and optionally documents
        include_docs: Whether to include retrieved documents in prompt

    Returns:
        Formatted prompt string
    """
    event = question["target_event"]
    options = [
        f"A. {question['option_A']}",
        f"B. {question['option_B']}",
        f"C. {question['option_C']}",
        f"D. {question['option_D']}",
    ]
    options_text = "\n".join(options)

    context = ""
    if include_docs and question.get("documents"):
        docs_text = "\n\n".join(
            f"Document {i+1}: {doc.get('content', doc.get('snippet', ''))[:500]}..."
            for i, doc in enumerate(question["documents"][:3])
        )
        context = f"\n\nRelevant Context:\n{docs_text}\n"

    prompt = f"""Given the following event, identify the most plausible direct cause(s) from the options below.
Note: There may be one or more correct answers.
{context}
Event: {event}

Options:
{options_text}

Which option(s) represent the most plausible direct cause(s) of the event?
Respond with only the letter(s) of your answer, separated by commas if multiple (e.g., "A" or "A,B")."""

    return prompt


def chain_of_thought_prompt(question: dict, include_docs: bool = False) -> str:
    """
    Generate a chain-of-thought prompt for cause identification.

    Args:
        question: Question dict with target_event, options, and optionally documents
        include_docs: Whether to include retrieved documents in prompt

    Returns:
        Formatted prompt string
    """
    event = question["target_event"]
    options = [
        f"A. {question['option_A']}",
        f"B. {question['option_B']}",
        f"C. {question['option_C']}",
        f"D. {question['option_D']}",
    ]
    options_text = "\n".join(options)

    context = ""
    if include_docs and question.get("documents"):
        docs_text = "\n\n".join(
            f"Document {i+1}: {doc.get('content', doc.get('snippet', ''))[:500]}..."
            for i, doc in enumerate(question["documents"][:3])
        )
        context = f"\n\nRelevant Context:\n{docs_text}\n"

    prompt = f"""Given the following event, identify the most plausible direct cause(s) from the options below.
Note: There may be one or more correct answers.
{context}
Event: {event}

Options:
{options_text}

Let's think step by step:
1. First, understand what the event describes
2. For each option, analyze if it could directly cause this event
3. Consider temporal and logical plausibility
4. Identify which cause(s) have the strongest direct causal link
5. Note that "None of the others are correct causes" is valid if other options don't fit

After your analysis, provide your final answer as: "Answer: [letter(s)]"
Use comma separation for multiple answers (e.g., "Answer: A,B")"""

    return prompt


def parse_response(response: str) -> set[str]:
    """
    Parse model response to extract predicted answer labels.

    Args:
        response: Model's text response

    Returns:
        Set of predicted labels (e.g., {"A", "B"})
    """
    import re

    response = response.upper()

    # Try to find "Answer: X" or "Answer: X,Y" pattern
    match = re.search(r"ANSWER:\s*([A-D](?:\s*,\s*[A-D])*)", response)
    if match:
        labels = match.group(1).replace(" ", "").split(",")
        return set(labels)

    # Try to find standalone letters at the end
    match = re.search(r"([A-D](?:\s*,\s*[A-D])*)\s*$", response)
    if match:
        labels = match.group(1).replace(" ", "").split(",")
        return set(labels)

    # Look for any A, B, C, D mentions
    found = set(re.findall(r"\b([A-D])\b", response))
    if found:
        return found

    # Default to A if parsing fails
    return {"A"}


def score_prediction(prediction: set[str], gold: set[str]) -> float:
    """
    Score a single prediction against gold labels.

    Scoring:
    - 1.0: Perfect match (P == G)
    - 0.5: Partial match (P is non-empty proper subset of G)
    - 0.0: Otherwise (incorrect options or empty prediction)

    Args:
        prediction: Set of predicted labels
        gold: Set of gold labels

    Returns:
        Score (0.0, 0.5, or 1.0)
    """
    if prediction == gold:
        return 1.0
    elif prediction and prediction.issubset(gold):
        return 0.5
    else:
        return 0.0


def evaluate_predictions(
    predictions: list[set[str]],
    golds: list[set[str]]
) -> dict[str, Any]:
    """
    Evaluate predictions against gold labels.

    Args:
        predictions: List of predicted label sets
        golds: List of gold label sets

    Returns:
        Dictionary of evaluation metrics
    """
    assert len(predictions) == len(golds)

    scores = [score_prediction(p, g) for p, g in zip(predictions, golds)]

    full_match = sum(1 for s in scores if s == 1.0)
    partial_match = sum(1 for s in scores if s == 0.5)
    incorrect = sum(1 for s in scores if s == 0.0)

    return {
        "average_score": sum(scores) / len(scores),
        "full_match": full_match,
        "partial_match": partial_match,
        "incorrect": incorrect,
        "total": len(scores),
        "full_match_rate": full_match / len(scores),
        "partial_match_rate": partial_match / len(scores),
        "incorrect_rate": incorrect / len(scores),
    }
