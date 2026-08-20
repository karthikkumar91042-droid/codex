"""
Knuth-Morris-Pratt (KMP) Substring Pattern Matching Algorithm.
Provides deterministic O(N + M) exact pattern search with context snippet extraction.
"""
from typing import List, Dict, Any


def compute_lps_array(pattern: str) -> List[int]:
    """
    Computes the Longest Prefix Suffix (LPS) array for pattern preprocessing.
    LPS[i] stores the length of the longest proper prefix of pattern[0..i]
    that is also a suffix of pattern[0..i].
    """
    length = 0
    m = len(pattern)
    lps = [0] * m
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(text: str, pattern: str, case_sensitive: bool = False) -> List[int]:
    """
    Searches for occurrences of `pattern` in `text` using KMP Algorithm.
    
    Returns:
        List of 0-based starting indices where pattern occurs in text.
    """
    if not text or not pattern:
        return []

    search_text = text if case_sensitive else text.lower()
    search_pattern = pattern if case_sensitive else pattern.lower()

    n = len(search_text)
    m = len(search_pattern)

    if m > n:
        return []

    lps = compute_lps_array(search_pattern)
    indices = []

    i = 0  # index for search_text
    j = 0  # index for search_pattern

    while (n - i) >= (m - j):
        if search_pattern[j] == search_text[i]:
            i += 1
            j += 1

        if j == m:
            indices.append(i - j)
            j = lps[j - 1]
        elif i < n and search_pattern[j] != search_text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return indices


def find_pattern_with_context(
    text: str,
    pattern: str,
    context_window: int = 40,
    case_sensitive: bool = False
) -> List[Dict[str, Any]]:
    """
    Finds all occurrences of pattern in text and returns indices along with contextual snippets.
    """
    indices = kmp_search(text, pattern, case_sensitive=case_sensitive)
    results = []

    for idx in indices:
        start = max(0, idx - context_window)
        end = min(len(text), idx + len(pattern) + context_window)
        
        prefix = ("..." if start > 0 else "") + text[start:idx]
        matched_str = text[idx:idx + len(pattern)]
        suffix = text[idx + len(pattern):end] + ("..." if end < len(text) else "")
        
        results.append({
            "index": idx,
            "pattern": pattern,
            "matched_text": matched_str,
            "context": f"{prefix}[{matched_str}]{suffix}",
            "prefix": prefix,
            "suffix": suffix
        })
    return results


def find_multiple_keywords_kmp(
    text: str,
    keywords: List[str],
    case_sensitive: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Executes KMP search for a list of candidate keywords/skills against the text.
    
    Returns:
        Dictionary mapping keyword -> {'count': int, 'indices': list, 'occurrences': list}
    """
    matches = {}
    for kw in keywords:
        kw_clean = kw.strip()
        if not kw_clean:
            continue
        occurrences = find_pattern_with_context(text, kw_clean, case_sensitive=case_sensitive)
        if occurrences:
            matches[kw_clean] = {
                "count": len(occurrences),
                "indices": [occ["index"] for occ in occurrences],
                "occurrences": occurrences
            }
    return matches
