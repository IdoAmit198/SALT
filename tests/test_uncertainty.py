import math
from types import SimpleNamespace

import pytest

from salt_benchmark.evaluation import (
    AUROC,
    ECE_calc,
    auroc_score,
    evaluate_atomic_response,
    evaluate_uncertainty_scores,
    prr_from_correctness,
)
from salt_benchmark.generation import (
    CONFIDENCE_SPECS,
    compute_confidence_from_logprob_vectors,
    extract_fenced_logprob_vectors,
    normalize_generation_logprobs,
    unit_confidence_scores_from_logprobs,
)
from salt_benchmark.tasks import MatrixMultiplicationProblem
from salt_benchmark.evaluation import zscore_sigmoid_calibration


def test_toy_matrix_atom_uncertainty_metrics_match_expected_values():
    evaluation = evaluate_atomic_response(
        "********\n5 4\n6 7 8\n********",
        "5 4\n5 7",
        MatrixMultiplicationProblem(),
        unit="atom",
    )
    confidences = [1.0, 1.0, 0.8, 0.8, 0.8]
    result = evaluate_uncertainty_scores(evaluation, confidences)
    assert result.correctness == [1, 1, 0, 1, 0]
    assert result.auroc == pytest.approx(5 / 6)
    assert result.ece == pytest.approx(0.28)
    assert result.precision == pytest.approx(3 / 5)


def test_toy_matrix_line_and_generation_uncertainty_metrics_match_expected_values():
    problem = MatrixMultiplicationProblem()
    line_evaluation = evaluate_atomic_response("********\n5 4\n6 7 8\n********", "5 4\n5 7", problem, unit="line")
    line_result = evaluate_uncertainty_scores(line_evaluation, [1.0, 0.8])
    assert line_result.correctness == [1, 0]
    assert line_result.auroc == pytest.approx(1.0)
    assert line_result.ece == pytest.approx(0.4)

    generation_evaluation = evaluate_atomic_response(
        "********\n5 4\n6 7 8\n********",
        "5 4\n5 7",
        problem,
        unit="generation",
    )
    generation_result = evaluate_uncertainty_scores(generation_evaluation, [0.88])
    assert generation_result.correctness == [0]
    assert generation_result.auroc is None
    assert generation_result.ece == pytest.approx(0.88)


def test_legacy_metric_wrappers_follow_source_benchmark_behavior():
    confidence = [1.0, 1.0, 0.8, 0.8, 0.8]
    correctness = [1, 1, 0, 1, 0]
    ece, mce = ECE_calc(confidence, correctness)
    assert AUROC(confidence, correctness) == pytest.approx(5 / 6)
    assert auroc_score([0.88], [0]) is None
    assert ece == pytest.approx(0.28)
    assert mce == pytest.approx(0.8 - (1 / 3))


def test_prr_port_matches_original_sanity_case():
    result = prr_from_correctness([False, False, True, True], [0.10, 0.20, 0.90, 0.80])
    assert result.prr == pytest.approx(1.0)
    assert result.base_error == pytest.approx(0.5)


def test_confidence_functions_are_pure_python_and_include_legacy_aliases():
    metrics = compute_confidence_from_logprob_vectors([[0.0, -1.0], [-1.0, -2.0]])
    assert metrics["max_probability"] >= metrics["median_probability"]
    for legacy_key in ("mean_prob", "median_prob", "max_prob", "our_max_prob", "ppl", "our_ppl"):
        assert legacy_key not in metrics
    assert metrics["num_tokens"] == 2
    assert CONFIDENCE_SPECS["mean_probability"]["higher_is_confident"] is True
    assert CONFIDENCE_SPECS["perplexity"]["higher_is_confident"] is False


def test_uncertainty_style_score_direction_is_used_for_ranking_metrics():
    evaluation = evaluate_atomic_response(
        "********\n5 4\n6 7\n********",
        "5 4\n5 7",
        MatrixMultiplicationProblem(),
        unit="atom",
    )
    perplexities = [1.0, 1.5, 8.0, 2.0]
    calibrated = zscore_sigmoid_calibration(perplexities, perplexities, higher_is_confident=False)

    result = evaluate_uncertainty_scores(
        evaluation,
        perplexities,
        score_name="perplexity",
        calibration_confidences=calibrated,
    )
    assert result.correctness == [1, 1, 0, 1]
    assert result.ranking_confidences == [-1.0, -1.5, -8.0, -2.0]
    assert result.auroc == pytest.approx(1.0)


def test_zscore_sigmoid_calibration_uses_train_distribution_and_direction():
    confidence_style = zscore_sigmoid_calibration([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    uncertainty_style = zscore_sigmoid_calibration([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], higher_is_confident=False)

    assert confidence_style[0] < confidence_style[1] < confidence_style[2]
    assert uncertainty_style[0] > uncertainty_style[1] > uncertainty_style[2]
    assert all(0.0 < value < 1.0 for value in confidence_style + uncertainty_style)


def test_logprob_normalization_supports_vllm_like_objects_and_unit_scoring():
    logprobs = [
        {1: SimpleNamespace(logprob=0.0, decoded_token="********\n")},
        {2: SimpleNamespace(logprob=0.0, decoded_token="5")},
        {3: SimpleNamespace(logprob=0.0, decoded_token=" ")},
        {4: SimpleNamespace(logprob=0.0, decoded_token="4")},
        {5: SimpleNamespace(logprob=0.0, decoded_token="\n")},
        {6: SimpleNamespace(logprob=-1.0, decoded_token="6")},
        {7: SimpleNamespace(logprob=-1.0, decoded_token=" ")},
        {8: SimpleNamespace(logprob=-1.0, decoded_token="7")},
        {9: SimpleNamespace(logprob=-1.0, decoded_token=" ")},
        {10: SimpleNamespace(logprob=-1.0, decoded_token="8")},
        {11: SimpleNamespace(logprob=0.0, decoded_token="\n********")},
    ]
    vectors = normalize_generation_logprobs(logprobs)
    assert vectors[0].token == "********\n"
    clean_response, aligned = extract_fenced_logprob_vectors("********\n5 4\n6 7 8\n********", logprobs)
    assert clean_response == "5 4\n6 7 8"
    assert aligned

    scores = unit_confidence_scores_from_logprobs(
        "********\n5 4\n6 7 8\n********",
        logprobs,
        MatrixMultiplicationProblem(),
        unit="atom",
    )
    assert len(scores) == 5
    assert all(math.isfinite(score) for score in scores)
