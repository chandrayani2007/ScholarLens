import re


def normalize_title(title: str) -> str:
    """
    Normalize paper title for deduplication matching:
    1. Convert to lowercase.
    2. Strip non-alphanumeric characters (keep alphanumeric and spaces).
    3. Collapse multiple whitespace into single space.
    """
    if not title:
        return ""

    # Convert to lowercase
    lowered = title.lower()

    # Keep letters, digits, and spaces
    alphanumeric_only = re.sub(r"[^a-z0-9\s]", " ", lowered)

    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", alphanumeric_only).strip()

    return normalized


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity ratio between two strings.
    Returns a float between 0.0 (completely different) and 1.0 (exact match).
    """
    str1 = normalize_title(s1)
    str2 = normalize_title(s2)

    if str1 == str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    len1, len2 = len(str1), len(str2)
    # Dynamic programming matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    distance = dp[len1][len2]
    max_len = max(len1, len2)

    return 1.0 - (distance / max_len)
