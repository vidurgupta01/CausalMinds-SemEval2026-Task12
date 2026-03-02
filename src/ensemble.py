"""
Ensemble and persona-based approaches for SemEval 2026 Task 12.
"""

from typing import Any
from collections import Counter


# ============================================================================
# PERSONA PROMPTS - Different expert perspectives
# ============================================================================

PERSONAS = {
    "historian": {
        "system": "You are a historian specializing in contemporary events and their causes. You analyze events through the lens of historical causation - what directly led to what.",
        "strength": ["politics", "elections", "policy", "war", "diplomatic"],
    },
    "journalist": {
        "system": "You are an investigative journalist who traces events back to their root causes. You focus on the 5 W's - especially WHY things happened.",
        "strength": ["breaking news", "scandal", "announcement", "statement", "reported"],
    },
    "economist": {
        "system": "You are an economist analyzing market events and their causes. You understand how economic forces, policies, and market psychology drive events.",
        "strength": ["market", "price", "stock", "trade", "tariff", "economic", "financial", "bank"],
    },
    "political_analyst": {
        "system": "You are a political analyst who understands how political actions lead to consequences. You trace policy decisions to their effects.",
        "strength": ["impeach", "vote", "parliament", "congress", "president", "election", "law", "bill"],
    },
    "crisis_expert": {
        "system": "You are a crisis management expert who analyzes emergency situations and their triggers. You identify what actions or events precipitated a crisis.",
        "strength": ["emergency", "disaster", "crisis", "protest", "riot", "attack", "death", "killed"],
    },
}


def detect_domain(question: dict) -> str:
    """Detect the domain/topic of a question to select best persona."""
    text = (question["target_event"] + " " +
            question["option_A"] + " " +
            question["option_B"] + " " +
            question["option_C"] + " " +
            question["option_D"]).lower()

    scores = {}
    for persona, info in PERSONAS.items():
        score = sum(1 for keyword in info["strength"] if keyword in text)
        scores[persona] = score

    # Return persona with highest score, default to journalist
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "journalist"


def make_persona_prompt(question: dict, persona: str) -> tuple[str, str]:
    """Create a prompt with persona system message."""
    info = PERSONAS.get(persona, PERSONAS["journalist"])

    user_prompt = f"""Analyze this event and identify its DIRECT CAUSE(s).

Event: {question['target_event']}

Options:
A. {question['option_A']}
B. {question['option_B']}
C. {question['option_C']}
D. {question['option_D']}

Which option(s) DIRECTLY CAUSED this event? Consider:
- Temporal order (cause must come before effect)
- Direct causal mechanism (not just correlation)
- Multiple causes are possible

Answer with letter(s) only:"""

    return info["system"], user_prompt


# ============================================================================
# ENSEMBLE PROMPTS - Multiple perspectives, then vote
# ============================================================================

ENSEMBLE_PROMPTS = {
    "temporal": """Focus on TEMPORAL ORDER. Which option(s) happened BEFORE the event and could have caused it?

Event: {event}
A. {A}
B. {B}
C. {C}
D. {D}

Options that happened BEFORE and caused the event (letters only):""",

    "mechanism": """Focus on CAUSAL MECHANISM. Which option(s) have a clear cause-effect relationship with the event?

Event: {event}
A. {A}
B. {B}
C. {C}
D. {D}

Options with clear causal mechanism (letters only):""",

    "elimination": """Use ELIMINATION. First identify options that are clearly NOT causes (effects, unrelated, or temporally after). Then select what remains.

Event: {event}
A. {A}
B. {B}
C. {C}
D. {D}

After eliminating non-causes, the direct cause(s) are (letters only):""",

    "counterfactual": """Use COUNTERFACTUAL reasoning. Ask: "If this option hadn't happened, would the event still occur?"

Event: {event}
A. {A}
B. {B}
C. {C}
D. {D}

Options where removing them would prevent the event (letters only):""",
}


def make_ensemble_prompts(question: dict) -> list[str]:
    """Create multiple prompts for ensemble voting."""
    prompts = []
    for name, template in ENSEMBLE_PROMPTS.items():
        prompt = template.format(
            event=question["target_event"],
            A=question["option_A"],
            B=question["option_B"],
            C=question["option_C"],
            D=question["option_D"],
        )
        prompts.append(prompt)
    return prompts


def ensemble_vote(predictions: list[set[str]], weights: list[float] = None) -> set[str]:
    """
    Aggregate predictions using weighted voting.
    An option is included if it gets >= 50% of votes.
    """
    if weights is None:
        weights = [1.0] * len(predictions)

    vote_counts = Counter()
    total_weight = sum(weights)

    for pred, weight in zip(predictions, weights):
        for label in pred:
            vote_counts[label] += weight

    # Include options with >= 50% support
    threshold = total_weight * 0.5
    result = {label for label, count in vote_counts.items() if count >= threshold}

    # If nothing passes threshold, take highest voted option
    if not result and vote_counts:
        result = {vote_counts.most_common(1)[0][0]}

    return result if result else {"A"}


# ============================================================================
# ROUTER - Select best approach based on question characteristics
# ============================================================================

def analyze_question(question: dict) -> dict[str, Any]:
    """Analyze question characteristics for routing."""
    text = question["target_event"].lower()
    options = [question[f"option_{l}"] for l in "ABCD"]

    has_none = any("none of the other" in opt.lower() for opt in options)
    has_temporal_words = any(w in text for w in ["after", "following", "then", "subsequently"])
    domain = detect_domain(question)

    # Count how many options mention the event's key terms
    event_words = set(text.split()) - {"the", "a", "an", "of", "in", "to", "and", "was", "were", "is", "are"}
    option_relevance = [
        sum(1 for w in event_words if w in opt.lower())
        for opt in options
    ]
    high_similarity = max(option_relevance) > 3  # Many shared words = potential distractor

    return {
        "has_none_option": has_none,
        "has_temporal_words": has_temporal_words,
        "domain": domain,
        "high_similarity": high_similarity,
        "option_relevance": option_relevance,
    }


def route_strategy(question: dict) -> str:
    """Decide which strategy to use based on question analysis."""
    analysis = analyze_question(question)

    # If high similarity between options and event, use ensemble (more robust to distractors)
    if analysis["high_similarity"]:
        return "ensemble"

    # If has "none" option, use elimination strategy
    if analysis["has_none_option"]:
        return "elimination"

    # Default: use persona based on domain
    return f"persona:{analysis['domain']}"
