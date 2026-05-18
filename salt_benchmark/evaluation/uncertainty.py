from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from salt_benchmark.generation.confidence import CONFIDENCE_SPECS


@dataclass(frozen=True)
class RejectionCurve:
    """A prediction rejection curve ordered from rejected to retained examples."""

    rejection_fraction: list[float]
    coverage: list[float]
    overall_error: list[float]
    selective_risk: list[float]
    order: list[int]


@dataclass(frozen=True)
class PRRResult:
    """Prediction Rejection Ratio and the areas used to compute it."""

    prr: float
    ar_uns: float
    ar_orc: float
    area_model: float
    area_random: float
    area_oracle: float
    base_error: float
    model_curve: RejectionCurve
    oracle_curve: RejectionCurve


@dataclass(frozen=True)
class UncertaintyEvaluationResult:
    """Precision, calibration, ranking, and rejection metrics for one granularity."""

    precision: float
    ece: float
    mce: float
    auroc: float | None
    prr: float
    confidences: list[float]
    correctness: list[int]
    ranking_confidences: list[float]


def AUROC(confidence: Sequence[float], ground_truth: Sequence[int | bool], sort: bool = True) -> float:
    """Compute AUROC with the pair-counting implementation used in the source benchmark.

    This is a pure-Python port of ``AUROC`` from
    ``benchmark_hallucinations/utils/uncertainty_estimation.py``. It preserves
    the original smoothing behavior: if there are no comparable pairs, the
    returned value is ``0.0``.
    """
    confidences, correctness = _validate_confidence_and_ground_truth(confidence, ground_truth)
    samples = list(zip(confidences, correctness))
    if sort:
        samples = sorted(enumerate(samples), key=lambda item: (-item[1][0], item[0]))
        samples = [sample for _, sample in samples]

    total_samples = len(samples)
    incorrect_after_me = [0] * total_samples
    for index in range(total_samples - 1, -1, -1):
        if index == total_samples - 1:
            incorrect_after_me[index] = 0
        else:
            incorrect_after_me[index] = incorrect_after_me[index + 1] + (1 - samples[index + 1][1])

    concordant = 0
    discordant = 0
    incorrect_before_me = 0
    for index, (_, is_correct) in enumerate(samples):
        if index != 0:
            incorrect_before_me += 1 - samples[index - 1][1]
        if is_correct:
            concordant += incorrect_after_me[index]
            discordant += incorrect_before_me
        else:
            discordant += (total_samples - index - 1) - incorrect_after_me[index]
            concordant += index - incorrect_before_me

    smoothing = 0.000001 if (concordant + discordant) == 0 else 0.0
    return concordant / (concordant + discordant + smoothing)


def auroc_score(confidence: Sequence[float], ground_truth: Sequence[int | bool], sort: bool = True) -> float | None:
    """Compute AUROC, returning None when all labels belong to one class."""
    _, correctness = _validate_confidence_and_ground_truth(confidence, ground_truth)
    if len(set(correctness)) < 2:
        return None
    return AUROC(confidence, ground_truth, sort=sort)


def ECE_calc(
    confidence: Sequence[float],
    ground_truth: Sequence[int | bool],
    num_bins: int = 15,
    bin_boundaries_scheme: Any = None,
) -> tuple[float, float]:
    """Compute expected and maximum calibration error.

    This is a pure-Python port of ``ECE_calc`` from the source benchmark. Bins
    are open on the lower bound and closed on the upper bound, matching the
    original implementation and the paper notation ``((j-1)/m, j/m]``.
    """
    if bin_boundaries_scheme is not None:
        raise ValueError("Custom bin boundary schemes are not supported by the pure-Python SALT port.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    confidences, correctness = _validate_confidence_and_ground_truth(confidence, ground_truth)
    _validate_probability_values(confidences, "confidence")
    bin_boundaries = [index / num_bins for index in range(num_bins + 1)]
    max_calibration_error = 0.0
    bins_accumulated_error = 0.0

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        bin_items = [
            (conf_value, correct_value)
            for conf_value, correct_value in zip(confidences, correctness)
            if conf_value <= upper and conf_value > lower
        ]
        if not bin_items:
            continue
        bin_avg_confidence = sum(item[0] for item in bin_items) / len(bin_items)
        bin_avg_accuracy = sum(item[1] for item in bin_items) / len(bin_items)
        bin_error = abs(bin_avg_confidence - bin_avg_accuracy)
        max_calibration_error = max(max_calibration_error, bin_error)
        bins_accumulated_error += bin_error * len(bin_items)

    return bins_accumulated_error / len(confidences), max_calibration_error


def expected_calibration_error(confidence: Sequence[float], ground_truth: Sequence[int | bool], num_bins: int = 15) -> float:
    """Return only the expected calibration error component of ``ECE_calc``."""
    ece, _ = ECE_calc(confidence, ground_truth, num_bins=num_bins)
    return ece


def rejection_curve_from_order(
    losses: Sequence[float],
    order: Sequence[int],
    weights: Sequence[float] | None = None,
) -> RejectionCurve:
    """Build a rejection curve from a given rejection order."""
    losses_arr = _validate_float_sequence(losses, "losses")
    order_arr = [int(index) for index in order]
    if len(order_arr) != len(losses_arr):
        raise ValueError("order must contain exactly one position for each loss.")
    if sorted(order_arr) != list(range(len(losses_arr))):
        raise ValueError("order must be a permutation of range(n).")
    weights_arr = _validate_weights(weights, len(losses_arr))

    total_weight = sum(weights_arr)
    if total_weight <= 0:
        raise ValueError("The sum of weights must be positive.")

    base_loss_mass = sum(weight * loss for weight, loss in zip(weights_arr, losses_arr))
    rejected_weight = 0.0
    rejected_loss_mass = 0.0
    rejection_fraction = [0.0]
    coverage = [1.0]
    overall_error = [base_loss_mass / total_weight]
    selective_risk = [overall_error[0]]

    for index in order_arr:
        rejected_weight += weights_arr[index]
        rejected_loss_mass += weights_arr[index] * losses_arr[index]
        current_rejection = rejected_weight / total_weight
        current_coverage = 1.0 - current_rejection
        remaining_loss_mass = base_loss_mass - rejected_loss_mass
        current_overall_error = remaining_loss_mass / total_weight
        rejection_fraction.append(current_rejection)
        coverage.append(current_coverage)
        overall_error.append(current_overall_error)
        selective_risk.append(current_overall_error / current_coverage if current_coverage > 0 else math.nan)

    return RejectionCurve(
        rejection_fraction=rejection_fraction,
        coverage=coverage,
        overall_error=overall_error,
        selective_risk=selective_risk,
        order=order_arr,
    )


def prr_from_scores(
    losses: Sequence[float],
    scores: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    higher_score_more_confident: bool = True,
) -> PRRResult:
    """Compute Prediction Rejection Ratio, ported from the source benchmark."""
    losses_arr, scores_arr, weights_arr = _validate_inputs(losses, scores, weights)
    uncertainty = [-score for score in scores_arr] if higher_score_more_confident else list(scores_arr)
    model_order = sorted(range(len(uncertainty)), key=lambda index: (-uncertainty[index], index))
    oracle_order = sorted(range(len(losses_arr)), key=lambda index: (-losses_arr[index], index))
    model_curve = rejection_curve_from_order(losses_arr, model_order, weights_arr)
    oracle_curve = rejection_curve_from_order(losses_arr, oracle_order, weights_arr)
    total_weight = sum(weights_arr)
    base_error = sum(loss * weight for loss, weight in zip(losses_arr, weights_arr)) / total_weight
    random_error = [base_error * (1.0 - fraction) for fraction in model_curve.rejection_fraction]
    area_random = _trapezoid(random_error, model_curve.rejection_fraction)
    area_model = _trapezoid(model_curve.overall_error, model_curve.rejection_fraction)
    area_oracle = _trapezoid(oracle_curve.overall_error, oracle_curve.rejection_fraction)
    ar_uns = area_random - area_model
    ar_orc = area_random - area_oracle
    prr = math.nan if math.isclose(ar_orc, 0.0) else ar_uns / ar_orc
    return PRRResult(
        prr=prr,
        ar_uns=ar_uns,
        ar_orc=ar_orc,
        area_model=area_model,
        area_random=area_random,
        area_oracle=area_oracle,
        base_error=base_error,
        model_curve=model_curve,
        oracle_curve=oracle_curve,
    )


def prr_from_correctness(
    correct: Sequence[bool | int],
    scores: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    higher_score_more_confident: bool = True,
) -> PRRResult:
    """Convenience wrapper for binary correctness labels."""
    losses = [1.0 - (1.0 if value else 0.0) for value in correct]
    return prr_from_scores(losses, scores, weights=weights, higher_score_more_confident=higher_score_more_confident)


def mean_group_prr_from_correctness(
    correct: Sequence[bool | int],
    scores: Sequence[float],
    group_ids: Sequence[object],
    *,
    weights: Sequence[float] | None = None,
    higher_score_more_confident: bool = True,
    group_weighting: str = "uniform",
) -> dict[str, object]:
    """Compute PRR per group and average finite group results."""
    if len(correct) != len(scores) or len(correct) != len(group_ids):
        raise ValueError("correct, scores, and group_ids must have the same length.")
    weights_arr = _validate_weights(weights, len(correct))
    per_group: dict[object, PRRResult] = {}
    prrs: list[float] = []
    aggregate_weights: list[float] = []
    unique_groups = []
    for group_id in group_ids:
        if group_id not in unique_groups:
            unique_groups.append(group_id)

    for group_id in unique_groups:
        indices = [index for index, current in enumerate(group_ids) if current == group_id]
        result = prr_from_correctness(
            [correct[index] for index in indices],
            [scores[index] for index in indices],
            weights=[weights_arr[index] for index in indices],
            higher_score_more_confident=higher_score_more_confident,
        )
        per_group[group_id] = result
        if math.isfinite(result.prr):
            prrs.append(result.prr)
            aggregate_weights.append(sum(weights_arr[index] for index in indices))

    if not prrs:
        mean_prr = math.nan
    elif group_weighting == "uniform":
        mean_prr = sum(prrs) / len(prrs)
    elif group_weighting == "weight":
        total = sum(aggregate_weights)
        mean_prr = sum(prr * weight for prr, weight in zip(prrs, aggregate_weights)) / total
    else:
        raise ValueError("group_weighting must be 'uniform' or 'weight'.")

    return {"mean_prr": mean_prr, "per_group": per_group}


def evaluate_uncertainty_scores(
    evaluation,
    confidences: Sequence[float],
    *,
    num_bins: int = 15,
    score_name: str | None = None,
    higher_is_confident: bool | None = None,
    calibration_confidences: Sequence[float] | None = None,
) -> UncertaintyEvaluationResult:
    """Evaluate confidence scores against generated-unit correctness labels.

    ``confidences`` may contain either confidence-style scores, where larger
    values mean more confidence, or uncertainty-style scores such as perplexity
    and entropy. Pass ``score_name`` for built-in logprob confidence functions,
    or ``higher_is_confident`` for custom scores, so AUROC and PRR use the right
    direction. ECE is computed from ``calibration_confidences`` when provided;
    otherwise it uses ``confidences`` and expects values in ``[0, 1]``.
    """
    generated_comparisons = [comparison for comparison in evaluation.comparisons if comparison.response_unit is not None]
    if len(generated_comparisons) != len(confidences):
        raise ValueError(
            f"Expected {len(generated_comparisons)} confidence scores for generated units, got {len(confidences)}."
        )
    correctness = [1 if comparison.is_correct else 0 for comparison in generated_comparisons]
    confidence_values = _validate_float_sequence(confidences, "confidences")
    confidence_direction = _resolve_confidence_direction(score_name, higher_is_confident)
    calibration_values = (
        _validate_float_sequence(calibration_confidences, "calibration_confidences")
        if calibration_confidences is not None
        else confidence_values
    )
    if len(calibration_values) != len(confidence_values):
        raise ValueError("calibration_confidences must have the same length as confidences.")
    ece, mce = ECE_calc(calibration_values, correctness, num_bins=num_bins)
    ranking_confidences = _ranking_confidences(confidence_values, confidence_direction)
    response_units = evaluation.metrics["num_response_units"]
    precision = evaluation.metrics["num_correct"] / response_units if response_units else 0.0
    prr = prr_from_correctness(correctness, ranking_confidences).prr
    return UncertaintyEvaluationResult(
        precision=precision,
        ece=ece,
        mce=mce,
        auroc=auroc_score(ranking_confidences, correctness),
        prr=prr,
        confidences=confidence_values,
        correctness=correctness,
        ranking_confidences=ranking_confidences,
    )


def zscore_sigmoid_calibration(
    train_scores: Sequence[float],
    scores: Sequence[float],
    *,
    higher_is_confident: bool = True,
    eps: float = 1e-12,
) -> list[float]:
    """Normalize scores with train-set z-scoring followed by a sigmoid.

    The returned values are confidence-style probabilities in ``[0, 1]``. For
    uncertainty-style inputs, pass ``higher_is_confident=False`` so lower raw
    scores map to higher calibrated confidence.
    """
    train_values = _validate_float_sequence(train_scores, "train_scores")
    score_values = _validate_float_sequence(scores, "scores")
    if not train_values:
        raise ValueError("At least one train score is required for calibration.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    mean_score = sum(train_values) / len(train_values)
    variance = sum((value - mean_score) ** 2 for value in train_values) / len(train_values)
    std = math.sqrt(max(variance, 0.0))
    if std <= eps:
        return [0.5 for _ in score_values]

    sign = 1.0 if higher_is_confident else -1.0
    return [_sigmoid(sign * ((value - mean_score) / std)) for value in score_values]


def _validate_confidence_and_ground_truth(
    confidence: Sequence[float],
    ground_truth: Sequence[int | bool],
) -> tuple[list[float], list[int]]:
    confidences = _validate_float_sequence(confidence, "confidence")
    correctness = [1 if value else 0 for value in ground_truth]
    if len(confidences) != len(correctness):
        raise ValueError("Confidence and correctness should have the same length.")
    if not confidences:
        raise ValueError("At least one sample is required.")
    return confidences, correctness


def _validate_inputs(
    losses: Sequence[float],
    scores: Sequence[float],
    weights: Sequence[float] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    losses_arr = _validate_float_sequence(losses, "losses")
    scores_arr = _validate_float_sequence(scores, "scores")
    if len(losses_arr) != len(scores_arr):
        raise ValueError(f"losses and scores must have the same length, got {len(losses_arr)} and {len(scores_arr)}.")
    if not losses_arr:
        raise ValueError("At least one example is required.")
    if any(loss < 0 for loss in losses_arr):
        raise ValueError("losses must be non-negative.")
    weights_arr = _validate_weights(weights, len(losses_arr))
    return losses_arr, scores_arr, weights_arr


def _validate_float_sequence(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contain non-finite values.")
    return result


def _validate_probability_values(values: Sequence[float], name: str) -> None:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} values used for calibration must be in [0, 1].")


def _resolve_confidence_direction(score_name: str | None, higher_is_confident: bool | None) -> bool:
    if score_name is not None:
        if score_name not in CONFIDENCE_SPECS:
            raise ValueError(f"Unknown confidence score name: {score_name}")
        spec_direction = bool(CONFIDENCE_SPECS[score_name]["higher_is_confident"])
        if higher_is_confident is not None and higher_is_confident != spec_direction:
            raise ValueError(f"score_name={score_name!r} implies higher_is_confident={spec_direction}.")
        return spec_direction
    return True if higher_is_confident is None else higher_is_confident


def _ranking_confidences(scores: Sequence[float], higher_is_confident: bool) -> list[float]:
    return list(scores) if higher_is_confident else [-score for score in scores]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _validate_weights(weights: Sequence[float] | None, expected_length: int) -> list[float]:
    if weights is None:
        return [1.0] * expected_length
    result = _validate_float_sequence(weights, "weights")
    if len(result) != expected_length:
        raise ValueError("weights must have the same length as the scored values.")
    if any(weight < 0 for weight in result):
        raise ValueError("weights must be non-negative.")
    if sum(result) <= 0:
        raise ValueError("The sum of weights must be positive.")
    return result


def _trapezoid(y_values: Sequence[float], x_values: Sequence[float]) -> float:
    if len(y_values) != len(x_values):
        raise ValueError("x and y sequences must have the same length.")
    total = 0.0
    for index in range(1, len(y_values)):
        total += (x_values[index] - x_values[index - 1]) * (y_values[index] + y_values[index - 1]) / 2.0
    return total
