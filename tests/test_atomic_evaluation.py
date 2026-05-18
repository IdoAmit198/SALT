from salt_benchmark.evaluation import (
    align_units_dynamic,
    compare_units,
    decompose_to_units,
    evaluate_atomic_response,
    fenced_response_extraction,
)
from salt_benchmark.tasks import DNAProblem, EditDistanceProblem, InsertProblem, LogicProblem, MatrixMultiplicationProblem, RotateProblem


def test_fenced_response_extraction_uses_last_valid_fence():
    response = "thinking\n********\nTCT CGA\n********\nnotes\n********\nanswer: bad\n********"
    fenced_text, start_index, end_index = fenced_response_extraction(response)
    assert fenced_text == "********\nTCT CGA\n********"
    assert start_index == response.index("********")
    assert end_index > start_index


def test_fenced_response_extraction_returns_empty_when_all_fences_are_invalid():
    response = "********\nfinal answer: TCT CGA\n********"
    assert fenced_response_extraction(response) == ("", -1, -1)


def test_decompose_to_atom_line_and_generation_units():
    assert decompose_to_units("********\nTCT CGA\n********", DNAProblem(), unit="atom") == ["TCT", "CGA"]
    assert decompose_to_units("p1: True\np2: False", LogicProblem(), unit="line") == ["p1: True", "p2: False"]
    assert decompose_to_units("1 2\n3 4", MatrixMultiplicationProblem(), unit="generation") == ["1 2\n3 4"]


def test_code_task_atom_decomposition_uses_original_regex_boundaries():
    assert decompose_to_units("3", EditDistanceProblem(), unit="atom") == ["3"]
    assert decompose_to_units("[[3, 1], [-4, -2]]", RotateProblem(), unit="atom") == ["[3, 1]", "[-4, -2]"]
    assert decompose_to_units("[[1, 5], [6, 9]]", InsertProblem(), unit="atom") == ["[1, 5]", "[6, 9]"]
    assert decompose_to_units("[<DK>, [7, 12]]", InsertProblem(), unit="atom") == ["<DK>", "[7, 12]"]


def test_evaluate_atomic_response_compares_positionally():
    result = evaluate_atomic_response(
        "scratch\n********\nTCT AAA\n********",
        "TCT CGA",
        DNAProblem(),
        unit="atom",
    )
    assert result.response_units == ["TCT", "AAA"]
    assert result.reference_units == ["TCT", "CGA"]
    assert [comparison.status for comparison in result.comparisons] == ["correct", "incorrect"]
    assert result.metrics["num_correct"] == 1
    assert result.metrics["precision"] == 0.5
    assert result.metrics["recall"] == 0.5
    assert result.metrics["accuracy"] == 0.5


def test_compare_units_distinguishes_recall_from_slot_coverage():
    _, metrics = compare_units(["5", "4", "6", "7", "8"], ["5", "4", "5", "7"], problem=MatrixMultiplicationProblem())
    assert metrics["num_correct"] == 3
    assert metrics["precision"] == 3 / 5
    assert metrics["recall"] == 3 / 4
    assert metrics["slot_coverage"] == 1.0
    assert metrics["coverage"] == metrics["slot_coverage"]


def test_compare_units_reports_missing_and_redundant_units():
    comparisons, metrics = compare_units(["A", "B", "C"], ["A", "B"], problem=DNAProblem())
    assert [comparison.status for comparison in comparisons] == ["correct", "correct", "redundant"]
    assert metrics["num_redundant"] == 1
    assert metrics["num_extra"] == 1
    assert metrics["hallucination_score"] == 0.5


def test_dynamic_alignment_recovers_after_omitted_unit():
    problem = MatrixMultiplicationProblem()
    strict = evaluate_atomic_response("********\n4 5 7\n********", "5 4\n5 7", problem, unit="atom")
    aligned = evaluate_atomic_response(
        "********\n4 5 7\n********",
        "5 4\n5 7",
        problem,
        unit="atom",
        dynamic_alignment=True,
    )

    assert align_units_dynamic(["4", "5", "7"], ["5", "4", "5", "7"], problem=problem) == [
        (None, "5"),
        ("4", "4"),
        ("5", "5"),
        ("7", "7"),
    ]
    assert [comparison.status for comparison in strict.comparisons] == ["incorrect", "incorrect", "incorrect", "missing"]
    assert strict.metrics["recall"] == 0.0
    assert [comparison.status for comparison in aligned.comparisons] == ["missing", "correct", "correct", "correct"]
    assert aligned.metrics["precision"] == 3 / 4
    assert aligned.metrics["recall"] == 3 / 4
    assert aligned.metrics["num_internal_missing"] == 1
    assert aligned.metrics["num_precision_units"] == 4
    assert aligned.metrics["dynamic_alignment"] is True


def test_dynamic_alignment_trailing_missing_does_not_reduce_precision():
    result = evaluate_atomic_response(
        "********\nTCT CGA TTC\n********",
        "TCT CGA TTC CGG",
        DNAProblem(),
        unit="atom",
        dynamic_alignment=True,
    )

    assert [comparison.status for comparison in result.comparisons] == ["correct", "correct", "correct", "missing"]
    assert result.metrics["precision"] == 1.0
    assert result.metrics["recall"] == 3 / 4
    assert result.metrics["num_internal_missing"] == 0
    assert result.metrics["num_precision_units"] == 3