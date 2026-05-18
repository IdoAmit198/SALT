import random
from pathlib import Path

from datasets import load_from_disk

from salt_benchmark.datasets import (
    create_code_dataset,
    create_dna_dataset,
    create_kronecker_product_dataset,
    create_logic_dataset,
    create_matrix_multiplication_dataset,
    create_needles_in_haystack_dataset,
)
from salt_benchmark.tasks.dna import complementary_dna
from salt_benchmark.tasks.matrix import parse_two_matrices, solve_kronecker_product, solve_matrix_multiplication
from salt_benchmark.tasks.needles import find_occurrences, generate_needles_sample

ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_DATASETS = {
    "DNA_dataset": {"task", "input", "reference"},
    "Code_dataset": {"task", "input", "reference"},
    "Logic_dataset": {"task", "input", "reference"},
    "Matrix_multiplication_dataset": {"task", "input", "reference"},
    "Kronecker_product_dataset": {"task", "input", "reference"},
    "NeedlesInHaystack_dataset": {"task", "input", "subject", "reference"},
}


def test_reproduction_datasets_load():
    for dataset_name, expected_columns in REPRODUCTION_DATASETS.items():
        dataset = load_from_disk(str(ROOT / "data" / "reproduction" / dataset_name))
        assert expected_columns.issubset(set(dataset.column_names))
        assert len(dataset) > 0


def test_dna_dataset_schema_and_references():
    dataset = create_dna_dataset(num_samples=5, max_codons=4, seed=7)
    assert dataset.column_names == ["task", "input", "reference"]
    assert dataset[0]["task"] == "DNA Translation"
    for row in dataset:
        assert row["reference"] == complementary_dna(row["input"])


def test_save_flag_controls_persistence(tmp_path):
    unused_path = tmp_path / "unused"
    create_dna_dataset(num_samples=1, seed=1, save=False, save_dir=str(unused_path))
    assert not unused_path.exists()

    save_path = tmp_path / "dna"
    create_dna_dataset(num_samples=1, seed=1, save=True, save_dir=str(save_path))
    loaded = load_from_disk(str(save_path))
    assert len(loaded) == 1


def test_logic_dataset_is_deterministic():
    first = create_logic_dataset(num_samples=2, num_variables_range=(4, 4), num_initial_assignments=2, seed=5)
    second = create_logic_dataset(num_samples=2, num_variables_range=(4, 4), num_initial_assignments=2, seed=5)
    assert first.to_list() == second.to_list()
    assert first[0]["task"] == "First-Order Logic"
    assert "Initial Assignments:" in first[0]["input"]
    assert first[0]["reference"].strip()


def test_matrix_multiplication_references():
    dataset = create_matrix_multiplication_dataset(num_samples=3, output_lengths=[6], seed=11)
    for row in dataset:
        assert row["task"] == "Matrix Multiplication"
        assert row["reference"] == solve_matrix_multiplication(row["input"])


def test_matrix_multiplication_can_create_2x2_inputs():
    dataset = create_matrix_multiplication_dataset(
        num_samples=1,
        output_lengths=[4],
        min_inner_dim=2,
        max_inner_dim=2,
        seed=13,
    )
    matrix_a, matrix_b = parse_two_matrices(dataset[0]["input"])
    assert matrix_a.shape == (2, 2)
    assert matrix_b.shape == (2, 2)
    assert dataset[0]["reference"] == solve_matrix_multiplication(dataset[0]["input"])


def test_kronecker_references():
    dataset = create_kronecker_product_dataset(num_samples=3, output_lengths=[6], seed=11)
    for row in dataset:
        assert row["task"] == "Kronecker Product"
        assert row["reference"] == solve_kronecker_product(row["input"])


def test_code_dataset_columns_and_sample_count():
    dataset = create_code_dataset(num_samples=9, seed=2)
    assert dataset.column_names == ["task", "variant", "input", "reference"]
    assert len(dataset) == 9
    assert set(dataset["task"]) == {"Code"}
    assert set(dataset["variant"]).issubset(
        {
            "Two Sum",
            "Edit Distance",
            "Rotate Matrix",
            "Insert Interval",
            "Next Permutation",
            "Reverse Words",
            "Rotate Array",
        }
    )


def test_needles_dataset_subject_and_reference():
    dataset = create_needles_in_haystack_dataset(num_samples=5, seed=3)
    assert dataset.column_names == ["task", "input", "subject", "reference"]
    for row in dataset:
        assert row["task"] == "Words Collection"
        assert row["reference"] == find_occurrences(row["input"], row["subject"])


def test_capital_letters_needles_keep_lowercase_filler_out_of_reference():
    rng = random.Random(2)
    for _ in range(100):
        text, subject, reference = generate_needles_sample(rng, min_occurrences=3, max_occurrences=3)
        if subject == "capital_letters":
            break
    else:
        raise AssertionError("Expected to sample a capital_letters example.")

    assert reference == find_occurrences(text, subject)
    assert len(reference.split()) == 3
    assert sum(1 for character in text if character.isupper()) == 3
