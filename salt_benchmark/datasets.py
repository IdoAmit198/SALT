from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from datasets import Dataset

from .tasks.base import maybe_save_dataset
from .tasks.code import code_rows
from .tasks.dna import complementary_dna, generate_dna_sequence
from .tasks.logic import create_logic_sample
from .tasks.matrix import generate_kronecker_product_problem, generate_matrix_multiplication_problem
from .tasks.needles import generate_needles_sample


def create_dna_dataset(
    num_samples: int = 100,
    min_codons: int = 1,
    max_codons: int = 30,
    seed: int | None = 42,
    save: bool = False,
    save_dir: str | None = None,
) -> Dataset:
    """Create a HuggingFace Dataset for the DNA Translation task.

    Args:
        num_samples: Number of examples to generate.
        min_codons: Minimum number of codons in each input sequence.
        max_codons: Maximum number of codons in each input sequence.
        seed: Random seed for reproducible sampling. Use None for non-deterministic sampling.
        save: Whether to save the dataset with ``Dataset.save_to_disk``.
        save_dir: Output directory used when ``save`` is True.

    Returns:
        A Dataset with ``task``, ``input``, and ``reference`` columns.
    """
    rng = random.Random(seed)
    inputs = [generate_dna_sequence(rng, min_codons=min_codons, max_codons=max_codons) for _ in range(num_samples)]
    references = [complementary_dna(sequence) for sequence in inputs]
    dataset = Dataset.from_dict({"task": ["DNA Translation"] * num_samples, "input": inputs, "reference": references})
    return maybe_save_dataset(dataset, save=save, save_dir=save_dir)


def create_logic_dataset(
    num_samples: int = 100,
    num_variables_range: tuple[int, int] = (8, 12),
    num_initial_assignments: int = 3,
    seed: int | None = 42,
    save: bool = False,
    save_dir: str | None = None,
) -> Dataset:
    """Create a HuggingFace Dataset for First-Order Logic deduction.

    Args:
        num_samples: Number of examples to generate.
        num_variables_range: Inclusive ``(min_variables, max_variables)`` range.
        num_initial_assignments: Number of truth assignments shown in the input.
        seed: Random seed for reproducible sampling. Use None for non-deterministic sampling.
        save: Whether to save the dataset with ``Dataset.save_to_disk``.
        save_dir: Output directory used when ``save`` is True.

    Returns:
        A Dataset with ``task``, ``input``, and ``reference`` columns.
    """
    min_variables, max_variables = num_variables_range
    if min_variables < 2 or max_variables < min_variables:
        raise ValueError("Expected 2 <= min_variables <= max_variables.")
    rng = random.Random(seed)
    inputs: list[str] = []
    references: list[str] = []
    for _ in range(num_samples):
        num_variables = rng.randrange(min_variables, max_variables + 1, 2) if min_variables != max_variables else min_variables
        initial_assignments = min(num_initial_assignments, max(1, num_variables - 1))
        sample_input, sample_reference = create_logic_sample(num_variables, initial_assignments, rng=rng)
        inputs.append(sample_input)
        references.append(sample_reference)
    dataset = Dataset.from_dict({"task": ["First-Order Logic"] * num_samples, "input": inputs, "reference": references})
    return maybe_save_dataset(dataset, save=save, save_dir=save_dir)


def create_matrix_multiplication_dataset(
    num_samples: int = 100,
    output_lengths: Sequence[int] | None = None,
    min_inner_dim: int = 2,
    max_inner_dim: int = 3,
    seed: int | None = 42,
    allow_scalar_operations: bool = False,
    save: bool = False,
    save_dir: str | None = None,
) -> Dataset:
    """Create a HuggingFace Dataset for matrix multiplication.

    Args:
        num_samples: Number of examples to generate.
        output_lengths: Candidate numbers of scalar entries in the output matrix.
        min_inner_dim: Minimum shared dimension between matrix A and matrix B.
        max_inner_dim: Maximum shared dimension between matrix A and matrix B.
        seed: Random seed for reproducible sampling. Use None for non-deterministic sampling.
        allow_scalar_operations: Whether to allow scalar-shaped matrix operands.
        save: Whether to save the dataset with ``Dataset.save_to_disk``.
        save_dir: Output directory used when ``save`` is True.

    Returns:
        A Dataset with ``task``, ``input``, and ``reference`` columns.
    """
    rng = np.random.default_rng(seed)
    lengths = list(output_lengths or range(6, 66, 2))
    inputs, references = _create_matrix_rows(
        num_samples=num_samples,
        lengths=lengths,
        generator=lambda length: generate_matrix_multiplication_problem(
            length,
            rng=rng,
            allow_scalar_operations=allow_scalar_operations,
            min_inner_dim=min_inner_dim,
            max_inner_dim=max_inner_dim,
        ),
    )
    dataset = Dataset.from_dict({"task": ["Matrix Multiplication"] * num_samples, "input": inputs, "reference": references})
    return maybe_save_dataset(dataset, save=save, save_dir=save_dir)


def create_kronecker_product_dataset(
    num_samples: int = 100,
    output_lengths: Sequence[int] | None = None,
    seed: int | None = 42,
    allow_scalar_operations: bool = False,
    save: bool = False,
    save_dir: str | None = None,
) -> Dataset:
    """Create a HuggingFace Dataset for Kronecker products.

    Args:
        num_samples: Number of examples to generate.
        output_lengths: Candidate numbers of scalar entries in the output matrix.
        seed: Random seed for reproducible sampling. Use None for non-deterministic sampling.
        allow_scalar_operations: Whether to allow scalar-shaped matrix operands.
        save: Whether to save the dataset with ``Dataset.save_to_disk``.
        save_dir: Output directory used when ``save`` is True.

    Returns:
        A Dataset with ``task``, ``input``, and ``reference`` columns.
    """
    rng = np.random.default_rng(seed)
    lengths = list(output_lengths or range(6, 66, 2))
    inputs, references = _create_matrix_rows(
        num_samples=num_samples,
        lengths=lengths,
        generator=lambda length: generate_kronecker_product_problem(
            length,
            rng=rng,
            allow_scalar_operations=allow_scalar_operations,
        ),
    )
    dataset = Dataset.from_dict({"task": ["Kronecker Product"] * num_samples, "input": inputs, "reference": references})
    return maybe_save_dataset(dataset, save=save, save_dir=save_dir)


def _create_matrix_rows(num_samples: int, lengths: Sequence[int], generator) -> tuple[list[str], list[str]]:
    """Generate matrix task rows, skipping candidate shapes the task cannot use."""
    if not lengths:
        raise ValueError("At least one output length must be provided.")
    inputs: list[str] = []
    references: list[str] = []
    attempts = 0
    max_attempts = max(100, num_samples * 20)
    while len(inputs) < num_samples and attempts < max_attempts:
        attempts += 1
        length = lengths[(attempts - 1) % len(lengths)]
        try:
            sample_input, sample_reference = generator(length)
        except ValueError:
            continue
        inputs.append(sample_input)
        references.append(sample_reference)
    if len(inputs) < num_samples:
        raise RuntimeError("Could not generate enough matrix samples from the requested output lengths.")
    return inputs, references


def create_code_dataset(
    num_samples: int = 14,
    seed: int | None = 42,
    save: bool = False,
    save_dir: str | None = None,
) -> Dataset:
    """Create a small HuggingFace Dataset of deterministic code-output tasks.

    Args:
        num_samples: Number of examples to return, cycling through built-in examples if needed.
        seed: Random seed used to shuffle the built-in examples.
        save: Whether to save the dataset with ``Dataset.save_to_disk``.
        save_dir: Output directory used when ``save`` is True.

    Returns:
        A Dataset with ``task``, ``input``, and ``reference`` columns.
    """
    rows = code_rows(num_samples)
    rng = random.Random(seed)
    rng.shuffle(rows)
    dataset = Dataset.from_dict(
        {
            "task": [row["task"] for row in rows],
            "variant": [row["variant"] for row in rows],
            "input": [row["input"] for row in rows],
            "reference": [row["reference"] for row in rows],
        }
    )
    return maybe_save_dataset(dataset, save=save, save_dir=save_dir)


def create_needles_in_haystack_dataset(
    num_samples: int = 100,
    min_occurrences: int = 2,
    max_occurrences: int = 6,
    seed: int | None = 42,
    save: bool = False,
    save_dir: str | None = None,
) -> Dataset:
    """Create a HuggingFace Dataset for the Words Collection task.

    Args:
        num_samples: Number of examples to generate.
        min_occurrences: Minimum number of target items inserted into each text.
        max_occurrences: Maximum number of target items inserted into each text.
        seed: Random seed for reproducible sampling. Use None for non-deterministic sampling.
        save: Whether to save the dataset with ``Dataset.save_to_disk``.
        save_dir: Output directory used when ``save`` is True.

    Returns:
        A Dataset with ``task``, ``input``, ``subject``, and ``reference`` columns.
    """
    rng = random.Random(seed)
    texts: list[str] = []
    subjects: list[str] = []
    references: list[str] = []
    for _ in range(num_samples):
        text, subject, reference = generate_needles_sample(
            rng,
            min_occurrences=min_occurrences,
            max_occurrences=max_occurrences,
        )
        texts.append(text)
        subjects.append(subject)
        references.append(reference)
    dataset = Dataset.from_dict(
        {
            "task": ["Words Collection"] * num_samples,
            "input": texts,
            "subject": subjects,
            "reference": references,
        }
    )
    return maybe_save_dataset(dataset, save=save, save_dir=save_dir)
