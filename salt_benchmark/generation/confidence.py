from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

DEFAULT_TOP_K = 20
DEFAULT_EPS = 1e-20
DEFAULT_SENTINEL_THRESHOLD = -1000.0
CONFIDENCE_SPECS = {
    "mean_probability": {"higher_is_confident": True, "needs_calibration": False},
    "median_probability": {"higher_is_confident": True, "needs_calibration": False},
    "max_probability": {"higher_is_confident": True, "needs_calibration": False},
    "log_probability_sum": {"higher_is_confident": True, "needs_calibration": True},
    "max_log_probability": {"higher_is_confident": True, "needs_calibration": True},
    "log_perplexity": {"higher_is_confident": False, "needs_calibration": True},
    "perplexity": {"higher_is_confident": False, "needs_calibration": True},
    "max_entropy": {"higher_is_confident": False, "needs_calibration": True},
    "mean_entropy": {"higher_is_confident": False, "needs_calibration": True},
}


def preprocess_logprob_vectors(
    token_logprob_vectors: Sequence[Sequence[float] | None],
    top_k: int = DEFAULT_TOP_K,
    eps: float = DEFAULT_EPS,
    sentinel_threshold: float = DEFAULT_SENTINEL_THRESHOLD,
) -> dict[str, Any]:
    """Normalize raw token logprob vectors and derive token probabilities."""
    _validate_top_k_and_eps(top_k, eps)
    token_prob_distributions: list[list[float]] = []
    greedy_log_likelihoods: list[float] = []
    entropies: list[float] = []

    for token_distribution in token_logprob_vectors:
        dist = list(token_distribution) if token_distribution is not None else []
        if not dist:
            dist = [0.0] * top_k
        if dist[0] < sentinel_threshold:
            finite_alternatives = [lp for lp in dist[1:] if math.isfinite(float(lp))]
            if finite_alternatives:
                dist[0] = min(finite_alternatives) - 1.0
            elif len(dist) > 1:
                dist[0] = min(dist[1:]) - 1.0
            else:
                dist[0] = -20.0

        dist = [float(value) for value in dist[:top_k]]
        if len(dist) < top_k:
            dist.extend([-1e9] * (top_k - len(dist)))
        probs = _softmax(dist)
        token_prob_distributions.append(probs)
        greedy_log_likelihoods.append(math.log(_clip(probs[0], eps, 1.0)))
        entropies.append(_entropy(probs))

    if not token_prob_distributions:
        raise ValueError("token_logprob_vectors produced zero tokens")
    return {
        "token_prob_distributions": token_prob_distributions,
        "greedy_log_likelihoods": greedy_log_likelihoods,
        "entropies": entropies,
    }


def preprocess_probability_distributions(
    token_prob_distributions: Sequence[Sequence[float]],
    top_k: int = DEFAULT_TOP_K,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Normalize token probability vectors and derive log-likelihood arrays."""
    _validate_top_k_and_eps(top_k, eps)
    normalized_distributions: list[list[float]] = []
    greedy_log_likelihoods: list[float] = []
    entropies: list[float] = []

    for token_distribution in token_prob_distributions:
        dist = list(token_distribution) if token_distribution is not None else []
        if not dist:
            probs = [1.0] + ([0.0] * (top_k - 1))
        else:
            values = [max(float(value), 0.0) for value in dist[:top_k]]
            if len(values) < top_k:
                values.extend([0.0] * (top_k - len(values)))
            total = sum(values)
            probs = ([1.0] + ([0.0] * (top_k - 1))) if total <= 0.0 else [value / total for value in values]
        normalized_distributions.append(probs)
        greedy_log_likelihoods.append(math.log(_clip(probs[0], eps, 1.0)))
        entropies.append(_entropy(probs))

    if not normalized_distributions:
        raise ValueError("token_prob_distributions produced zero tokens")
    return {
        "token_prob_distributions": normalized_distributions,
        "greedy_log_likelihoods": greedy_log_likelihoods,
        "entropies": entropies,
    }


def compute_confidence_functions(
    greedy_log_likelihoods: Sequence[float],
    entropies: Sequence[float],
    token_prob_distributions: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Compute logprob-based confidence functions for one unit."""
    log_likelihoods = [float(value) for value in greedy_log_likelihoods]
    entropy_values = [float(value) for value in entropies]
    if not log_likelihoods:
        raise ValueError("greedy_log_likelihoods is empty")
    if not entropy_values:
        raise ValueError("entropies is empty")
    selected_probabilities = [math.exp(value) for value in log_likelihoods]
    mean_probability = sum(selected_probabilities) / len(selected_probabilities)
    median_probability = _median(selected_probabilities)
    max_probability = max(selected_probabilities)
    log_probability_sum = sum(log_likelihoods)
    max_log_probability = max(log_likelihoods)
    log_perplexity = -sum(log_likelihoods) / len(log_likelihoods)
    perplexity = math.exp(log_perplexity)
    max_entropy = max(entropy_values)
    mean_entropy = sum(entropy_values) / len(entropy_values)

    return {
        "mean_probability": mean_probability,
        "median_probability": median_probability,
        "max_probability": max_probability,
        "log_probability_sum": log_probability_sum,
        "max_log_probability": max_log_probability,
        "log_perplexity": log_perplexity,
        "perplexity": perplexity,
        "max_entropy": max_entropy,
        "mean_entropy": mean_entropy,
        "num_tokens": len(log_likelihoods),
        "token_prob_distributions": [list(distribution) for distribution in token_prob_distributions],
    }


def compute_confidence_from_logprob_vectors(
    token_logprob_vectors: Sequence[Sequence[float] | None],
    top_k: int = DEFAULT_TOP_K,
    eps: float = DEFAULT_EPS,
    sentinel_threshold: float = DEFAULT_SENTINEL_THRESHOLD,
) -> dict[str, Any]:
    """Compute confidence functions directly from token top-logprob vectors."""
    preprocessed = preprocess_logprob_vectors(
        token_logprob_vectors=token_logprob_vectors,
        top_k=top_k,
        eps=eps,
        sentinel_threshold=sentinel_threshold,
    )
    return compute_confidence_functions(
        greedy_log_likelihoods=preprocessed["greedy_log_likelihoods"],
        entropies=preprocessed["entropies"],
        token_prob_distributions=preprocessed["token_prob_distributions"],
    )


def compute_confidence_from_prob_distributions(
    token_prob_distributions: Sequence[Sequence[float]],
    top_k: int = DEFAULT_TOP_K,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Compute confidence functions directly from token probability vectors."""
    preprocessed = preprocess_probability_distributions(token_prob_distributions, top_k=top_k, eps=eps)
    return compute_confidence_functions(
        greedy_log_likelihoods=preprocessed["greedy_log_likelihoods"],
        entropies=preprocessed["entropies"],
        token_prob_distributions=preprocessed["token_prob_distributions"],
    )


def compute_batch_confidence_from_logprobs(
    samples_token_logprob_vectors: Mapping[str, Sequence[Sequence[float] | None]],
    top_k: int = DEFAULT_TOP_K,
    eps: float = DEFAULT_EPS,
    sentinel_threshold: float = DEFAULT_SENTINEL_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Compute confidence functions for a mapping of sample IDs to token vectors."""
    return {
        sample_id: compute_confidence_from_logprob_vectors(
            token_logprob_vectors,
            top_k=top_k,
            eps=eps,
            sentinel_threshold=sentinel_threshold,
        )
        for sample_id, token_logprob_vectors in samples_token_logprob_vectors.items()
    }


def _validate_top_k_and_eps(top_k: int, eps: float) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if eps <= 0.0 or eps >= 1.0:
        raise ValueError("eps must satisfy 0 < eps < 1")


def _softmax(values: Sequence[float]) -> list[float]:
    max_value = max(values)
    exponentials = [math.exp(value - max_value) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
