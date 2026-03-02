"""
Optimized approaches for SemEval 2026 Task 12.
Uses training data patterns to improve performance.
"""

import json
import random
from pathlib import Path
from typing import Any


def load_training_examples(n: int = 5) -> list[dict]:
    """Load diverse training examples for few-shot prompting."""
    with open("data/official/train_data/questions.jsonl") as f:
        train = [json.loads(line) for line in f if line.strip()]

    # Get diverse examples: single answer, multi-answer, "None" cases
    single_ans = [q for q in train if "," not in q["golden_answer"]]
    multi_ans = [q for q in train if "," in q["golden_answer"]]
    none_cases = [q for q in train if "None" in q.get("option_C", "") or "None" in q.get("option_B", "")]

    examples = []
    if none_cases:
        examples.append(random.choice(none_cases))
    if multi_ans:
        examples.append(random.choice(multi_ans))

    # Fill rest with random single-answer examples
    remaining = n - len(examples)
    examples.extend(random.sample(single_ans, min(remaining, len(single_ans))))

    random.shuffle(examples)
    return examples[:n]


def format_example(q: dict) -> str:
    """Format a training example for few-shot prompt."""
    return f"""Event: {q['target_event']}
A. {q['option_A']}
B. {q['option_B']}
C. {q['option_C']}
D. {q['option_D']}
Answer: {q['golden_answer']}"""


def optimized_few_shot_prompt(question: dict, examples: list[dict]) -> str:
    """
    Optimized few-shot prompt with training insights.
    """
    examples_text = "\n\n".join(format_example(e) for e in examples)

    prompt = f"""You are an expert at causal reasoning. Given an event, identify which option(s) represent its most plausible DIRECT cause.

IMPORTANT RULES:
1. There may be ONE or MULTIPLE correct answers (about 43% of questions have multiple correct answers)
2. If an option says "None of the others are correct causes" - consider it carefully, it's often correct when other options are temporally/logically implausible
3. A direct cause must happen BEFORE the event and have a clear causal link
4. Avoid semantic distractors - options that are related but not actual causes

Here are some examples:

{examples_text}

Now answer this question:

Event: {question['target_event']}
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Think step by step:
1. What does the event describe?
2. For each option, is it temporally BEFORE the event?
3. Is there a direct causal link (not just correlation)?
4. Are multiple options valid direct causes?

Answer with ONLY the letter(s), comma-separated if multiple (e.g., "A" or "A,B"):"""

    return prompt


def optimized_with_verification_prompt(question: dict, examples: list[dict]) -> str:
    """
    Two-stage prompt: first identify candidates, then verify.
    """
    examples_text = "\n\n".join(format_example(e) for e in examples)

    prompt = f"""You are an expert at causal reasoning about real-world events.

TASK: Identify the most plausible DIRECT cause(s) of an event.

KEY INSIGHTS:
- ~43% of questions have MULTIPLE correct answers - don't default to single answer
- "None of the others are correct causes" is valid when other options aren't actual causes
- Direct cause = happens BEFORE the event + clear causal mechanism
- Beware of semantic distractors (related but not causal)

Examples:
{examples_text}

---

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

STEP 1 - For each option, determine if it could be a direct cause:
- A: [Could this directly cause the event? Yes/No + reason]
- B: [Could this directly cause the event? Yes/No + reason]
- C: [Could this directly cause the event? Yes/No + reason]
- D: [Could this directly cause the event? Yes/No + reason]

STEP 2 - Check for "None" option:
- If one option says "None of the others are correct" AND other options aren't valid causes, select it

STEP 3 - Final answer (letter(s) only, comma-separated if multiple):
Answer:"""

    return prompt


def self_consistency_prompt(question: dict) -> str:
    """Simple prompt for self-consistency (multiple samples)."""
    return f"""Identify the direct cause(s) of this event. There may be 1-4 correct answers.

Event: {question['target_event']}

A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Direct cause(s) (letter(s) only):"""


def parse_optimized_response(response: str) -> set[str]:
    """Parse response, handling various formats."""
    import re

    response = response.strip().upper()

    # Look for "Answer:" pattern first
    match = re.search(r'ANSWER[:\s]*([A-D](?:\s*,\s*[A-D])*)', response)
    if match:
        return set(match.group(1).replace(" ", "").split(","))

    # Look for pattern at end of response
    lines = response.strip().split('\n')
    last_line = lines[-1].strip()

    # Check if last line is just letters
    match = re.match(r'^([A-D](?:\s*,\s*[A-D])*)$', last_line)
    if match:
        return set(match.group(1).replace(" ", "").split(","))

    # Find all standalone letters
    found = set(re.findall(r'\b([A-D])\b', response[-100:]))  # Look at end
    if found:
        return found

    return {"A"}  # Default fallback


def aggregate_votes(predictions: list[set[str]], threshold: float = 0.5) -> set[str]:
    """
    Aggregate multiple predictions using voting.
    Include an option if it appears in >= threshold of predictions.
    """
    from collections import Counter

    vote_counts = Counter()
    for pred in predictions:
        for label in pred:
            vote_counts[label] += 1

    n = len(predictions)
    result = set()
    for label, count in vote_counts.items():
        if count / n >= threshold:
            result.add(label)

    return result if result else predictions[0]  # Fallback to first prediction
