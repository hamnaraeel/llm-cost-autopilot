"""Hand-built features for complexity classification.

Deliberately interpretable rather than learned embeddings: when the router
sends a request to a cheap model and the verifier later disagrees, you need to
be able to read the feature row and say *why* the classifier thought it was
simple. A 384-dim sentence embedding cannot give you that.

Every feature is cheap to compute -- the classifier sits in the request path,
so extraction has to cost microseconds, not milliseconds.
"""

from __future__ import annotations

import math
import re

# --- lexicons -------------------------------------------------------------
# Verbs that signal *what kind of work* the prompt is asking for. Grouped by
# the tier they most often indicate; overlap is fine, the model weighs them.

MECHANICAL_VERBS = {
    "extract", "list", "convert", "reformat", "format", "capitalize", "lowercase",
    "uppercase", "sort", "alphabetize", "count", "copy", "output", "return",
    "parse", "split", "join", "replace", "strip", "trim", "reorder", "transcribe",
}

CONDENSING_VERBS = {
    "summarize", "summarise", "classify", "categorize", "categorise", "label",
    "tag", "rewrite", "condense", "shorten", "paraphrase", "translate", "group",
    "rank", "score", "identify", "detect", "match",
}

REASONING_VERBS = {
    "analyze", "analyse", "compare", "contrast", "evaluate", "assess", "critique",
    "recommend", "justify", "weigh", "reason", "derive", "prove", "debug", "diagnose",
    "optimize", "optimise", "design", "architect", "decide", "plan", "troubleshoot",
    "refactor", "investigate", "reconcile", "forecast", "predict", "infer",
}

CREATIVE_VERBS = {
    "write", "draft", "compose", "invent", "imagine", "brainstorm", "generate",
    "craft", "pitch", "name",
}

# --- patterns -------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
_CODE_FENCE_RE = re.compile(r"```|~~~")
_CODE_HINT_RE = re.compile(
    r"\b(def |class |function |return |import |const |var |let |SELECT |=>|\{\s*$)", re.M
)
_CONTEXT_RE = re.compile(
    r"^\s*(context|background|given|source|document|text|input|passage|article|transcript)\s*:",
    re.I | re.M,
)
_QUOTED_BLOCK_RE = re.compile(r'"[^"]{60,}"|“[^”]{60,}”', re.S)
_LABEL_SET_RE = re.compile(
    r"(one of|exactly one of|choose from|categories?|labels?)\s*:?\s*[A-Z_]{2,}", re.I
)
_STEPWISE_RE = re.compile(
    r"(step[- ]by[- ]step|show your (reasoning|work)|explain your|walk through|"
    r"reason through|then state|first .{0,40}then)", re.I
)
_LENGTH_CONSTRAINT_RE = re.compile(
    r"(under|below|at most|no more than|fewer than|less than|exactly|within)\s+\d+\s*"
    r"(word|words|character|characters|sentence|sentences|line|lines|bullet|bullets)"
    r"|\b(one|two|three|four|five)\s+(sentence|sentences|word|words|line|lines|bullet|bullets)\b",
    re.I,
)
_NEGATIVE_CONSTRAINT_RE = re.compile(
    r"\b(do not|don't|never|without|must not|may not|avoid|none may|exclude)\b", re.I
)
_HARD_CONSTRAINT_RE = re.compile(r"\b(must|exactly|only|required|ensure|each|every)\b", re.I)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_ENUM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+", re.M)

_FORMAT_TOKENS = {
    "json": 3, "schema": 3, "yaml": 3, "xml": 3, "csv": 2, "table": 2,
    "markdown": 2, "bullet": 1, "bullets": 1, "list": 1, "one sentence": 1,
    "comma-separated": 1, "code block": 2,
}


def _count_verbs(words: set[str], lexicon: set[str]) -> int:
    return len(words & lexicon)


def extract_features(prompt: str) -> dict[str, float]:
    """Return the interpretable feature dict for one prompt."""
    text = prompt.strip()
    lower = text.lower()
    words = _WORD_RE.findall(lower)
    word_set = set(words)
    n_words = max(len(words), 1)

    # ~4 chars/token is close enough for a routing feature; the router does not
    # need tokenizer-exact counts, only a monotone size signal.
    est_tokens = len(text) / 4.0

    n_lines = text.count("\n") + 1
    n_sentences = max(len(_SENT_RE.findall(text)), 1)

    has_context = bool(_CONTEXT_RE.search(text)) or bool(_QUOTED_BLOCK_RE.search(text))
    # A long multi-line prompt is usually carrying material to work on.
    if n_lines >= 4 and len(text) > 300:
        has_context = True

    has_code = bool(_CODE_FENCE_RE.search(text)) or bool(_CODE_HINT_RE.search(text))

    format_score = 0
    for token, weight in _FORMAT_TOKENS.items():
        if token in lower:
            format_score = max(format_score, weight)

    n_constraints = (
        len(_HARD_CONSTRAINT_RE.findall(text))
        + len(_NEGATIVE_CONSTRAINT_RE.findall(text))
        + len(_LENGTH_CONSTRAINT_RE.findall(text))
    )

    numbers = _NUMBER_RE.findall(text)

    return {
        "log_chars": math.log1p(len(text)),
        "log_est_tokens": math.log1p(est_tokens),
        "n_words": float(n_words),
        "n_sentences": float(n_sentences),
        "n_lines": float(n_lines),
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "mechanical_verbs": float(_count_verbs(word_set, MECHANICAL_VERBS)),
        "condensing_verbs": float(_count_verbs(word_set, CONDENSING_VERBS)),
        "reasoning_verbs": float(_count_verbs(word_set, REASONING_VERBS)),
        "creative_verbs": float(_count_verbs(word_set, CREATIVE_VERBS)),
        "has_context": float(has_context),
        "has_code": float(has_code),
        "has_label_set": float(bool(_LABEL_SET_RE.search(text))),
        "asks_stepwise": float(bool(_STEPWISE_RE.search(text))),
        "has_length_constraint": float(bool(_LENGTH_CONSTRAINT_RE.search(text))),
        "n_negative_constraints": float(len(_NEGATIVE_CONSTRAINT_RE.findall(text))),
        "n_constraints": float(n_constraints),
        "format_complexity": float(format_score),
        "n_numbers": float(len(numbers)),
        "number_density": len(numbers) / n_words,
        "n_enumerated_items": float(len(_ENUM_RE.findall(text))),
        "n_questions": float(text.count("?")),
        "why_or_how": float(bool(re.search(r"\b(why|how come|what would happen)\b", lower))),
    }


FEATURE_NAMES: list[str] = list(extract_features("seed").keys())


def feature_vector(prompt: str) -> list[float]:
    """Feature values in a stable order -- what sklearn actually consumes."""
    f = extract_features(prompt)
    return [f[name] for name in FEATURE_NAMES]
