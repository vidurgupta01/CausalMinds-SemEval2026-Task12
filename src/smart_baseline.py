"""
Smart baseline with rule-based post-processing.
Uses patterns discovered from training data analysis.
"""

import re
from typing import Any


def has_none_option(question: dict) -> tuple[bool, str]:
    """Check if any option is 'None of the above' type."""
    for opt in ["option_A", "option_B", "option_C", "option_D"]:
        text = question.get(opt, "").lower()
        if "none of the other" in text or "none of the above" in text:
            return True, opt[-1]  # Return the letter
    return False, None


def smart_prompt(question: dict) -> str:
    """
    Smart prompt that explicitly handles edge cases.
    """
    has_none, none_letter = has_none_option(question)

    none_instruction = ""
    if has_none:
        none_instruction = f"""
IMPORTANT: Option {none_letter} states "None of the others are correct causes."
This is a valid answer IF AND ONLY IF none of the other options are plausible direct causes.
Carefully evaluate each other option first before selecting {none_letter}."""

    prompt = f"""You are an expert at identifying causal relationships in real-world events.

TASK: Select the option(s) that represent the most plausible DIRECT cause of the given event.

RULES:
1. A direct cause must temporally PRECEDE the event
2. There must be a clear causal mechanism (not just correlation or association)
3. Multiple options can be correct (select all that apply)
4. About 43% of questions have multiple correct answers - don't default to single answer
{none_instruction}

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

For each option, briefly assess:
- Does it happen BEFORE the event? (temporal order)
- Is there a direct causal link? (mechanism)

Then provide your final answer as just the letter(s), comma-separated if multiple.

Answer:"""

    return prompt


def smart_prompt_v2(question: dict, examples: list[dict] = None) -> str:
    """
    Version 2: More structured reasoning with examples.
    """
    has_none, none_letter = has_none_option(question)

    examples_text = ""
    if examples:
        ex_strs = []
        for ex in examples[:3]:
            ex_strs.append(f"""Event: {ex['target_event']}
A. {ex['option_A']}
B. {ex['option_B']}
C. {ex['option_C']}
D. {ex['option_D']}
Correct: {ex['golden_answer']}""")
        examples_text = "Examples:\n" + "\n\n".join(ex_strs) + "\n\n---\n\n"

    prompt = f"""Identify the direct cause(s) of the event. Think carefully about causation vs correlation.

{examples_text}Event: {question['target_event']}

A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Quick analysis (1 line each):
A: [cause or not?]
B: [cause or not?]
C: [cause or not?]
D: [cause or not?]

Answer (letters only):"""

    return prompt


def apply_none_rule(question: dict, prediction: set[str]) -> set[str]:
    """
    Post-processing: If model is uncertain and there's a "None" option,
    check if it should be selected.
    """
    has_none, none_letter = has_none_option(question)

    if not has_none:
        return prediction

    # If prediction is empty or only contains the none letter, keep it
    if not prediction or prediction == {none_letter}:
        return prediction

    # If prediction contains none_letter AND other options, that's contradictory
    # Remove the none option
    if none_letter in prediction and len(prediction) > 1:
        prediction = prediction - {none_letter}

    return prediction


def confidence_based_selection(
    question: dict,
    response: str,
    prediction: set[str]
) -> set[str]:
    """
    Adjust prediction based on confidence signals in response.
    """
    response_lower = response.lower()

    # If response shows uncertainty, be more conservative
    uncertainty_signals = ["unclear", "uncertain", "possibly", "might be", "could be"]
    has_uncertainty = any(sig in response_lower for sig in uncertainty_signals)

    if has_uncertainty and len(prediction) > 2:
        # If uncertain and selecting many options, might be over-selecting
        # This is a heuristic - may need tuning
        pass

    return prediction


def ensemble_predict(predictions: list[set[str]], weights: list[float] = None) -> set[str]:
    """
    Ensemble multiple predictions with optional weights.
    """
    from collections import Counter

    if weights is None:
        weights = [1.0] * len(predictions)

    vote_counts = Counter()
    for pred, weight in zip(predictions, weights):
        for label in pred:
            vote_counts[label] += weight

    total_weight = sum(weights)
    threshold = total_weight * 0.4  # 40% threshold

    result = {label for label, count in vote_counts.items() if count >= threshold}

    return result if result else predictions[0]
