import re
from typing import List


STOP_WORDS = frozenset({
    # English
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "is", "it", "be", "as", "do", "no",
    "not", "are", "was", "were", "been", "has", "had", "have", "from",
    "this", "that", "if", "so", "we", "he", "she", "they", "you",
    "its", "my", "our", "your", "his", "her", "all", "can", "will",
    "may", "up", "out", "use", "also", "each", "which", "when",
    "what", "how", "than", "then", "into", "just", "over", "such",
    "only", "some", "more", "other", "about", "these", "those",
    "them", "there", "their", "would", "could", "should",
    # Code noise
    "none", "true", "false", "null", "self", "cls", "args", "kwargs",
    "return", "def", "class", "import", "from",
})

_CAMEL_SPLIT = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"
    r"|(?<=[A-Z])(?=[A-Z][a-z])"
)

_NON_ALNUM = re.compile(r"[^a-zA-Z0-9_]")


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 indexing / querying.

    1. Replace non-alphanumeric chars (except _) with spaces.
    2. Split on whitespace and underscores (snake_case).
    3. Sub-split on camelCase / PascalCase boundaries.
    4. Lowercase everything.
    5. Remove tokens of length <= 1 and stop-words.

    Args:
        text: raw text to tokenize.

    Returns:
        List of cleaned, lowercased tokens.
    """
    if not text:
        return []

    cleaned = _NON_ALNUM.sub(" ", text)

    raw_tokens = re.split(r"[\s_]+", cleaned)

    tokens: List[str] = []
    for raw in raw_tokens:
        if not raw:
            continue
        sub_parts = _CAMEL_SPLIT.split(raw)
        for part in sub_parts:
            lower = part.lower()
            if len(lower) > 1 and lower not in STOP_WORDS:
                tokens.append(lower)

    return tokens
