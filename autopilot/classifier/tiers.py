"""Complexity tier definitions.

These are the contract between the labeled dataset, the classifier, and the
routing map. They are written out in prose because the labels were applied by
hand and a future labeler needs the same rubric.
"""

from __future__ import annotations

from autopilot.schemas import ComplexityTier

TIER_DEFINITIONS: dict[ComplexityTier, dict] = {
    ComplexityTier.SIMPLE: {
        "name": "simple",
        "summary": "Mechanical transformation or lookup. The answer is present in the prompt.",
        "includes": [
            "reformatting (case, list -> CSV, JSON shape changes)",
            "field extraction from provided text",
            "basic Q&A answerable from supplied context",
            "single-label lookup with an explicit key",
        ],
        "excludes": [
            "anything requiring outside knowledge",
            "anything requiring a judgment the prompt does not determine",
        ],
        "tell": "A careful intern with no domain knowledge could do it correctly.",
    },
    ComplexityTier.MODERATE: {
        "name": "moderate",
        "summary": "Compression, categorization, or bounded analysis over given material.",
        "includes": [
            "summarization to a length or shape",
            "classification into a supplied label set",
            "structured analysis with a stated output format",
            "sentiment/intent/tone reading",
            "short rewrites that must preserve meaning",
        ],
        "excludes": [
            "open-ended reasoning chains",
            "problems where a wrong intermediate step changes the answer",
        ],
        "tell": "There is one defensible answer, but producing it requires reading and condensing.",
    },
    ComplexityTier.COMPLEX: {
        "name": "complex",
        "summary": "Multi-step reasoning, creative generation, or nuanced judgment.",
        "includes": [
            "arithmetic/logic chains where step 2 depends on step 1",
            "debugging and code correctness reasoning",
            "tradeoff analysis and recommendations",
            "creative writing under multiple hard constraints",
            "anything asking to compare, critique, design, or decide",
        ],
        "excludes": [],
        "tell": "A wrong early step silently corrupts the final answer.",
    },
}

TIER_BY_NAME = {v["name"]: k for k, v in TIER_DEFINITIONS.items()}


def describe(tier: ComplexityTier) -> str:
    d = TIER_DEFINITIONS[tier]
    return f"Tier {tier.value} ({d['name']}): {d['summary']}"
