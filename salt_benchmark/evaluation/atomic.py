from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import regex

UnitKind = Literal["atom", "line", "generation"]
COMPLETE_FENCE_PATTERN = r"(?<!\*)[\*]{4,}(?!\*)"

BAD_FENCED_KEYWORDS = [
    "the answer is",
    "final answer",
    "the final answer is:",
    "the output is:",
    "output:",
    "answer:",
    "final output:",
    "final:",
    "final answer is:",
    "cannot conclude",
    "can't conclude",
    "can not conclude",
    "therefore",
    "thus",
    "hence",
    "answer",
    "output",
    "final",
    "input",
    "these",
    "invalid",
    "with",
    "without",
    "end",
]

SELECTIVE_UNITS = {"DK", "<DK", "DK>", "<DK>"}


@dataclass(frozen=True)
class UnitComparison:
    """Comparison between one response unit and one reference unit."""

    index: int
    response_unit: str | None
    reference_unit: str | None
    is_correct: bool
    status: str


@dataclass(frozen=True)
class AtomicEvaluationResult:
    """Fenced extraction, decomposition, and comparison result for one response."""

    unit: UnitKind
    fenced_response: str
    fence_start_index: int
    fence_end_index: int
    clean_response: str
    response_units: list[str]
    reference_units: list[str]
    comparisons: list[UnitComparison]
    metrics: dict[str, float | int]


def validate_response_format_via_keywords(response: str) -> bool:
    """Return True when a fenced region does not contain known bad format keywords."""
    response_lower = response.lower()
    return not any(keyword in response_lower for keyword in BAD_FENCED_KEYWORDS)


def fenced_response_extraction(response: str | None) -> tuple[str, int, int]:
    """Extract the intended fenced answer and its character span.

    The search follows the original benchmark behavior: prefer the last valid fenced
    region, reject fenced regions that contain bad format keywords, and accept a
    trailing unclosed fence if it contains usable content.

    Returns:
        ``(text, start, end)``. If no valid fenced answer exists, returns ``("", -1, -1)``.
    """
    if response is None or response == "":
        return "", -1, -1

    clean_space_response = "\n".join(response.split("\n"))
    fence_matches = list(regex.finditer(COMPLETE_FENCE_PATTERN, clean_space_response, overlapped=True))

    if len(fence_matches) >= 2:
        # Pair consecutive complete fences. A broad overlapped region regex can
        # accidentally match the text between a closing fence and the next opening fence.
        pair_count = len(fence_matches) // 2
        for pair_index in range(pair_count - 1, -1, -1):
            opening = fence_matches[pair_index * 2]
            closing = fence_matches[pair_index * 2 + 1]
            matched_text = clean_space_response[opening.start() : closing.end()]
            if matched_text.strip() != "" and matched_text.replace("*", "").strip() == "":
                continue
            if not validate_response_format_via_keywords(matched_text):
                continue
            return matched_text, opening.start(), closing.end()

        if len(fence_matches) % 2 == 1:
            last_fence = fence_matches[-1]
            last_fenced_text = clean_space_response[last_fence.start() :]
            last_fenced_content = last_fenced_text.replace("*", "").strip()
            if last_fenced_content != "" and validate_response_format_via_keywords(last_fenced_text):
                return last_fenced_text, last_fence.start(), len(clean_space_response)

        return "", -1, -1

    if len(fence_matches) == 1:
        last_fence = fence_matches[-1]
        last_fenced_text = clean_space_response[last_fence.start() :]
        last_fenced_content = last_fenced_text.replace("*", "").strip()
        if last_fenced_content != "" and validate_response_format_via_keywords(last_fenced_text):
            return last_fenced_text, last_fence.start(), len(clean_space_response)
        return "", -1, -1

    return response, 0, len(response)


def strip_fence_markers(text: str) -> str:
    """Remove asterisk fence markers and surrounding whitespace from text."""
    return text.replace("*", "").strip()


def decompose_to_units(text: str, problem, unit: UnitKind = "atom") -> list[str]:
    """Split text into generation, line, or atom units.

    Atom mode uses ``problem.delimiter`` when present and falls back to
    ``problem.regex_exp`` for regex-based tasks.
    """
    clean_text = strip_fence_markers(text)
    if unit == "generation":
        return [clean_text] if clean_text else []

    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
    if unit == "line":
        return lines
    if unit != "atom":
        raise ValueError(f"Unsupported unit kind: {unit}")

    delimiter = getattr(problem, "delimiter", None)
    regex_exp = getattr(problem, "regex_exp", None)
    atoms: list[str] = []
    if delimiter:
        for line in lines:
            atoms.extend(atom.strip() for atom in line.split(delimiter) if atom.strip())
    elif regex_exp:
        for line in lines:
            atoms.extend(match.group(0).strip() for match in regex.finditer(regex_exp, line) if match.group(0).strip())
    else:
        raise ValueError("Expected problem.delimiter or problem.regex_exp for atom decomposition.")

    if atoms and atoms[0].startswith("[") and not atoms[0].endswith("]"):
        atoms[0] = atoms[0][1:]
    if atoms and atoms[-1].endswith("]") and not atoms[-1].startswith("["):
        atoms[-1] = atoms[-1][:-1]
    return atoms


def compare_units(
    response_units: list[str],
    reference_units: list[str],
    problem=None,
    *,
    dynamic_alignment: bool = False,
) -> tuple[list[UnitComparison], dict[str, float | int]]:
    """Compare response and reference units by strict position or dynamic alignment.

    Returns:
        A list of unit comparisons and a metrics dictionary containing
        precision, recall, slot coverage, redundant/missing counts, and hallucination score.
    """
    aligned_units = (
        align_units_dynamic(response_units, reference_units, problem=problem)
        if dynamic_alignment
        else _strict_aligned_units(response_units, reference_units)
    )
    comparisons: list[UnitComparison] = []
    correct = 0
    present_aligned_slots = 0
    false_negative = 0

    for index, (response_unit, reference_unit) in enumerate(aligned_units):
        if response_unit is None:
            status = "missing"
            is_correct = False
        elif reference_unit is None:
            status = "redundant"
            is_correct = False
        elif _is_selective(response_unit):
            status = "selective"
            is_correct = False
            present_aligned_slots += 1
            false_negative += 1
        else:
            present_aligned_slots += 1
            is_correct = _normalize_unit(response_unit, problem) == _normalize_unit(reference_unit, problem)
            status = "correct" if is_correct else "incorrect"
            correct += int(is_correct)
        comparisons.append(
            UnitComparison(
                index=index,
                response_unit=response_unit,
                reference_unit=reference_unit,
                is_correct=is_correct,
                status=status,
            )
        )

    reference_len = len(reference_units)
    response_len = len(response_units)
    extra = sum(1 for comparison in comparisons if comparison.status == "redundant")
    missing = sum(1 for comparison in comparisons if comparison.status == "missing")
    internal_missing = _count_internal_missing(comparisons) if dynamic_alignment else 0
    precision_denominator = response_len + internal_missing
    denominator_without_selective = max(reference_len - false_negative, 0)
    metrics = {
        "num_response_units": response_len,
        "num_reference_units": reference_len,
        "num_correct": correct,
        "num_missing": missing,
        "num_internal_missing": internal_missing,
        "num_precision_units": precision_denominator,
        "num_redundant": extra,
        "num_extra": extra,
        "num_present_aligned_slots": present_aligned_slots,
        "num_selective": false_negative,
        "dynamic_alignment": dynamic_alignment,
        "precision": correct / precision_denominator if precision_denominator else 0,
        "recall": correct / reference_len if reference_len else 0,
        "slot_coverage": present_aligned_slots / reference_len if reference_len else 0,
        "accuracy": correct / denominator_without_selective if denominator_without_selective else 0,
        "accuracy_include_extra": correct / (reference_len - false_negative + extra) if (reference_len - false_negative + extra) else 0,
        "coverage": present_aligned_slots / reference_len if reference_len else 0,
        "hallucination_score": extra / reference_len if reference_len else 0,
    }
    return comparisons, metrics


def align_units_dynamic(
    response_units: list[str],
    reference_units: list[str],
    problem=None,
    *,
    match_score: int = 2,
    mismatch_penalty: int = 1,
    gap_penalty: int = 1,
) -> list[tuple[str | None, str | None]]:
    """Globally align response and reference units with Needleman-Wunsch.

    Needleman-Wunsch is a classic dynamic programming algorithm for global
    sequence alignment. It fills a score table whose rows are reference prefixes
    and columns are response prefixes. Each cell stores the best score reachable
    by one of three moves: diagonal for aligning two units, up for aligning a
    reference unit to a gap, or left for aligning a response unit to a gap. A
    traceback from the bottom-right cell yields the aligned unit pairs.

    This simplified variant aligns SALT units rather than characters and returns
    ``None`` for gaps. Equal normalized units receive ``match_score``; unequal
    aligned units pay ``mismatch_penalty``; gaps pay ``gap_penalty``.
    """
    if match_score <= 0:
        raise ValueError("match_score must be positive.")
    if mismatch_penalty < 0 or gap_penalty < 0:
        raise ValueError("penalties must be non-negative.")

    reference_len = len(reference_units)
    response_len = len(response_units)
    scores = [[0] * (response_len + 1) for _ in range(reference_len + 1)]
    traceback = [[""] * (response_len + 1) for _ in range(reference_len + 1)]

    for ref_index in range(1, reference_len + 1):
        scores[ref_index][0] = scores[ref_index - 1][0] - gap_penalty
        traceback[ref_index][0] = "up"
    for response_index in range(1, response_len + 1):
        scores[0][response_index] = scores[0][response_index - 1] - gap_penalty
        traceback[0][response_index] = "left"

    for ref_index in range(1, reference_len + 1):
        for response_index in range(1, response_len + 1):
            response_unit = response_units[response_index - 1]
            reference_unit = reference_units[ref_index - 1]
            diagonal_delta = match_score if _units_match(response_unit, reference_unit, problem) else -mismatch_penalty
            candidates = [
                (scores[ref_index - 1][response_index - 1] + diagonal_delta, "diagonal"),
                (scores[ref_index - 1][response_index] - gap_penalty, "up"),
                (scores[ref_index][response_index - 1] - gap_penalty, "left"),
            ]
            scores[ref_index][response_index], traceback[ref_index][response_index] = max(
                candidates,
                key=lambda candidate: candidate[0],
            )

    aligned: list[tuple[str | None, str | None]] = []
    ref_index = reference_len
    response_index = response_len
    while ref_index > 0 or response_index > 0:
        move = traceback[ref_index][response_index]
        if move == "diagonal":
            aligned.append((response_units[response_index - 1], reference_units[ref_index - 1]))
            ref_index -= 1
            response_index -= 1
        elif move == "up":
            aligned.append((None, reference_units[ref_index - 1]))
            ref_index -= 1
        elif move == "left":
            aligned.append((response_units[response_index - 1], None))
            response_index -= 1
        else:
            raise RuntimeError("Needleman-Wunsch traceback reached an uninitialized cell.")

    aligned.reverse()
    return aligned


def evaluate_atomic_response(
    response: str | None,
    reference: str,
    problem,
    unit: UnitKind = "atom",
    *,
    dynamic_alignment: bool = False,
) -> AtomicEvaluationResult:
    """Extract, decompose, and evaluate one generated response against a reference."""
    fenced_response, start_index, end_index = fenced_response_extraction(response)
    clean_response = strip_fence_markers(fenced_response) if fenced_response else ""
    response_units = decompose_to_units(clean_response, problem, unit=unit) if clean_response else []
    reference_units = decompose_to_units(reference, problem, unit=unit)
    comparisons, metrics = compare_units(
        response_units,
        reference_units,
        problem=problem,
        dynamic_alignment=dynamic_alignment,
    )
    return AtomicEvaluationResult(
        unit=unit,
        fenced_response=fenced_response,
        fence_start_index=start_index,
        fence_end_index=end_index,
        clean_response=clean_response,
        response_units=response_units,
        reference_units=reference_units,
        comparisons=comparisons,
        metrics=metrics,
    )


def _normalize_unit(unit: str, problem=None) -> str:
    """Normalize one unit before exact comparison."""
    normalized = unit.replace("*", "").strip()
    delimiter = getattr(problem, "delimiter", None)
    if delimiter is None or delimiter != " ":
        normalized = normalized.replace(" ", "")
    return normalized


def _strict_aligned_units(response_units: list[str], reference_units: list[str]) -> list[tuple[str | None, str | None]]:
    max_len = max(len(response_units), len(reference_units))
    return [
        (
            response_units[index] if index < len(response_units) else None,
            reference_units[index] if index < len(reference_units) else None,
        )
        for index in range(max_len)
    ]


def _units_match(response_unit: str, reference_unit: str, problem=None) -> bool:
    return (not _is_selective(response_unit)) and _normalize_unit(response_unit, problem) == _normalize_unit(reference_unit, problem)


def _count_internal_missing(comparisons: list[UnitComparison]) -> int:
    """Count response gaps that occur before later generated units."""
    missing_before_later_response = 0
    has_later_response = False
    for comparison in reversed(comparisons):
        if comparison.response_unit is not None:
            has_later_response = True
        elif comparison.status == "missing" and has_later_response:
            missing_before_later_response += 1
    return missing_before_later_response


def _is_selective(unit: str) -> bool:
    """Return whether a unit is one of SALT's don't-know markers."""
    return unit.strip().upper() in SELECTIVE_UNITS
