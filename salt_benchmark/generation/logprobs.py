from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from salt_benchmark.evaluation.atomic import fenced_response_extraction, strip_fence_markers

from .confidence import compute_confidence_from_logprob_vectors


@dataclass(frozen=True)
class TokenLogprobVector:
    """Top-logprob vector for one generated token."""

    token: str
    logprobs: list[float]


@dataclass(frozen=True)
class AlignedTokenLogprobVector:
    """A token logprob vector aligned to character offsets in the clean answer."""

    token: str
    logprobs: list[float]
    start: int
    end: int


def normalize_generation_logprobs(logprobs: Any) -> list[TokenLogprobVector]:
    """Normalize common vLLM logprob shapes into token vectors."""
    if logprobs is None:
        return []
    vectors: list[TokenLogprobVector] = []
    for entry in logprobs:
        candidates = list(_iter_candidates(entry))
        if not candidates:
            continue
        token = candidates[0][1]
        vectors.append(TokenLogprobVector(token=token, logprobs=[candidate[0] for candidate in candidates]))
    return vectors


def extract_fenced_logprob_vectors(response_text: str, logprobs: Any) -> tuple[str, list[AlignedTokenLogprobVector]]:
    """Return the clean fenced answer and token logprobs overlapping that answer."""
    fenced_response, _, _ = fenced_response_extraction(response_text)
    clean_response = strip_fence_markers(fenced_response) if fenced_response else ""
    if not clean_response:
        return "", []
    vectors = normalize_generation_logprobs(logprobs)
    reconstructed = "".join(vector.token for vector in vectors)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for vector in vectors:
        spans.append((cursor, cursor + len(vector.token)))
        cursor += len(vector.token)

    start = reconstructed.find(clean_response)
    end = start + len(clean_response) if start >= 0 else -1
    if start < 0:
        normalized_match = _find_normalized_span(reconstructed, clean_response)
        if normalized_match is None:
            return clean_response, []
        start, end = normalized_match

    aligned: list[AlignedTokenLogprobVector] = []
    for vector, (token_start, token_end) in zip(vectors, spans):
        if token_start < end and token_end > start:
            aligned.append(
                AlignedTokenLogprobVector(
                    token=vector.token,
                    logprobs=vector.logprobs,
                    start=max(token_start, start) - start,
                    end=min(token_end, end) - start,
                )
            )
    return clean_response, aligned


def unit_confidence_scores_from_logprobs(
    response_text: str,
    logprobs: Any,
    problem,
    unit: str,
    confidence_key: str = "mean_probability",
) -> list[float]:
    """Map generated token logprobs to SALT units and compute one score per unit."""
    clean_response, aligned_tokens = extract_fenced_logprob_vectors(response_text, logprobs)
    if not clean_response or not aligned_tokens:
        return []
    unit_spans = _unit_spans(clean_response, problem, unit)
    scores: list[float] = []
    for _, unit_start, unit_end in unit_spans:
        token_vectors = [
            token.logprobs
            for token in aligned_tokens
            if token.start < unit_end and token.end > unit_start
        ]
        if not token_vectors:
            return []
        metrics = compute_confidence_from_logprob_vectors(token_vectors)
        if confidence_key not in metrics:
            raise ValueError(f"Unknown confidence key: {confidence_key}")
        scores.append(float(metrics[confidence_key]))
    return scores


def _iter_candidates(entry: Any):
    if entry is None:
        return
    if isinstance(entry, (list, tuple)):
        for item in entry:
            yield from _iter_candidates(item)
        return
    if isinstance(entry, dict):
        for key, value in entry.items():
            pair = _candidate_to_pair(key, value)
            if pair is not None:
                yield pair
        return
    pair = _candidate_to_pair(None, entry)
    if pair is not None:
        yield pair


def _candidate_to_pair(key: object, value: Any) -> tuple[float, str] | None:
    if isinstance(value, (int, float)):
        return float(value), str(key or "")
    if isinstance(value, (tuple, list)) and value:
        if len(value) >= 2 and isinstance(value[0], (int, float)):
            return float(value[0]), str(value[1])
    logprob = getattr(value, "logprob", None)
    if logprob is not None:
        token = getattr(value, "decoded_token", None)
        if token is None:
            token = getattr(value, "token", None)
        if token is None:
            token = str(key or "")
        return float(logprob), str(token)
    return None


def _find_normalized_span(text: str, target: str) -> tuple[int, int] | None:
    normalized_chars: list[str] = []
    normalized_to_original: list[int] = []
    for index, char in enumerate(text):
        if not char.isspace():
            normalized_chars.append(char)
            normalized_to_original.append(index)
    normalized_target = "".join(char for char in target if not char.isspace())
    normalized_text = "".join(normalized_chars)
    start = normalized_text.find(normalized_target)
    if start < 0:
        return None
    end = start + len(normalized_target) - 1
    return normalized_to_original[start], normalized_to_original[end] + 1


def _unit_spans(text: str, problem, unit: str) -> list[tuple[str, int, int]]:
    if unit == "generation":
        stripped = text.strip()
        start = text.find(stripped) if stripped else 0
        return [(stripped, start, start + len(stripped))] if stripped else []
    if unit == "line":
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            line_without_newline = raw_line.rstrip("\n\r")
            stripped = line_without_newline.strip()
            if stripped:
                leading = len(line_without_newline) - len(line_without_newline.lstrip())
                start = cursor + leading
                spans.append((stripped, start, start + len(stripped)))
            cursor += len(raw_line)
        return spans
    if unit != "atom":
        raise ValueError(f"Unsupported unit kind: {unit}")

    delimiter = getattr(problem, "delimiter", None)
    regex_exp = getattr(problem, "regex_exp", None)
    if delimiter and delimiter.isspace():
        return _non_whitespace_spans(text)
    if delimiter:
        return _delimiter_spans(text, delimiter)
    if regex_exp:
        import regex

        return [(match.group(0).strip(), match.start(), match.end()) for match in regex.finditer(regex_exp, text) if match.group(0).strip()]
    raise ValueError("Expected problem.delimiter or problem.regex_exp for atom decomposition.")


def _non_whitespace_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isspace():
            if start is not None:
                spans.append((text[start:index], start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append((text[start:], start, len(text)))
    return spans


def _delimiter_spans(text: str, delimiter: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in text.split(delimiter):
        stripped = piece.strip()
        if stripped:
            start = text.find(stripped, cursor)
            spans.append((stripped, start, start + len(stripped)))
        cursor += len(piece) + len(delimiter)
    return spans
