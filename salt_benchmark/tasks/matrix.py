from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import FENCE_INSTRUCTION, Problem

MATRIX_REGEX = r"([+-]?\d+(\.\d+)?)|(<DK>)|(<DK)|(DK>)|(DK)|(<dk>)|(<dk)|(dk>)|(dk)"


def _matrix_to_text(name: str, matrix: np.ndarray) -> str:
    return name + "=\n" + "\n".join(" ".join(map(str, row)) for row in matrix)


def _solution_to_text(matrix: np.ndarray) -> str:
    return "\n".join(" ".join(map(str, row)) for row in matrix)


def _valid_multiplication_shapes(output_length: int, allow_scalar_operations: bool = False) -> list[tuple[int, int]]:
    factors = [(i, output_length // i) for i in range(1, int(output_length ** 0.5) + 1) if output_length % i == 0]
    if not allow_scalar_operations:
        factors = [shape for shape in factors if shape[0] > 1 and shape[1] > 1]
    return factors


def generate_matrix_multiplication_problem(
    output_length: int,
    rng: np.random.Generator | None = None,
    allow_scalar_operations: bool = False,
    min_inner_dim: int = 2,
    max_inner_dim: int = 3,
    value_low: int = -9,
    value_high: int = 9,
) -> tuple[str, str]:
    """Generate one matrix multiplication input and row-wise text reference.

    ``output_length`` controls the number of entries in the output matrix; the
    shared inner dimension is sampled between ``min_inner_dim`` and ``max_inner_dim``.
    """
    rng = rng or np.random.default_rng()
    shapes = _valid_multiplication_shapes(output_length, allow_scalar_operations)
    if not shapes:
        raise ValueError(f"No valid non-scalar matrix multiplication shape for output length {output_length}.")
    m, n = shapes[int(rng.integers(len(shapes)))]
    inner_dim = int(rng.integers(min_inner_dim, max_inner_dim + 1))
    matrix_a = rng.integers(value_low, value_high + 1, size=(m, inner_dim))
    matrix_b = rng.integers(value_low, value_high + 1, size=(inner_dim, n))
    solution = np.dot(matrix_a, matrix_b)
    return f"{_matrix_to_text('A', matrix_a)}\n{_matrix_to_text('B', matrix_b)}", _solution_to_text(solution)


def _valid_kronecker_shapes(output_length: int, allow_scalar_operations: bool = False) -> list[tuple[int, int, int, int]]:
    factors = [(i, output_length // i) for i in range(1, output_length + 1) if output_length % i == 0]
    valid_shapes: list[tuple[int, int, int, int]] = []
    for output_rows, output_cols in factors:
        for rows_a in range(1, output_rows + 1):
            if output_rows % rows_a != 0:
                continue
            rows_b = output_rows // rows_a
            for cols_a in range(1, output_cols + 1):
                if output_cols % cols_a != 0:
                    continue
                cols_b = output_cols // cols_a
                if allow_scalar_operations or ((rows_a > 1 or cols_a > 1) and (rows_b > 1 or cols_b > 1)):
                    if allow_scalar_operations or not ((rows_a, cols_a) == (1, 1) or (rows_b, cols_b) == (1, 1)):
                        valid_shapes.append((rows_a, cols_a, rows_b, cols_b))
    return valid_shapes


def generate_kronecker_product_problem(
    output_length: int,
    rng: np.random.Generator | None = None,
    allow_scalar_operations: bool = False,
    prefer_2d_probability: float = 0.5,
    value_low: int = -9,
    value_high: int = 9,
) -> tuple[str, str]:
    """Generate one Kronecker product input and row-wise text reference."""
    rng = rng or np.random.default_rng()
    shapes = _valid_kronecker_shapes(output_length, allow_scalar_operations)
    if not shapes:
        raise ValueError(f"No valid non-scalar Kronecker shape for output length {output_length}.")
    if rng.random() < prefer_2d_probability:
        two_dimensional = [shape for shape in shapes if 1 not in shape]
        if two_dimensional:
            shapes = two_dimensional
    rows_a, cols_a, rows_b, cols_b = shapes[int(rng.integers(len(shapes)))]
    matrix_a = rng.integers(value_low, value_high + 1, size=(rows_a, cols_a))
    matrix_b = rng.integers(value_low, value_high + 1, size=(rows_b, cols_b))
    solution = np.kron(matrix_a, matrix_b)
    return f"{_matrix_to_text('A', matrix_a)}\n{_matrix_to_text('B', matrix_b)}", _solution_to_text(solution)


def parse_two_matrices(textual_input: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse SALT matrix input text into matrices A and B."""
    if "\nB=" not in textual_input:
        raise ValueError("Expected matrix input containing '\\nB='.")
    matrix_a_text, matrix_b_text = textual_input.split("\nB=", 1)
    matrix_a_text = matrix_a_text.removeprefix("A=\n")
    matrix_b_text = matrix_b_text.strip()
    matrix_a = np.array([[int(value) for value in row.split()] for row in matrix_a_text.splitlines() if row.strip()])
    matrix_b = np.array([[int(value) for value in row.split()] for row in matrix_b_text.splitlines() if row.strip()])
    return matrix_a, matrix_b


def solve_matrix_multiplication(textual_input: str) -> str:
    """Solve one SALT matrix multiplication input and return row-wise text."""
    matrix_a, matrix_b = parse_two_matrices(textual_input)
    return _solution_to_text(np.dot(matrix_a, matrix_b))


def solve_kronecker_product(textual_input: str) -> str:
    """Solve one SALT Kronecker product input and return row-wise text."""
    matrix_a, matrix_b = parse_two_matrices(textual_input)
    return _solution_to_text(np.kron(matrix_a, matrix_b))


MATRIX_FEW_SHOT = [
    ("A=\n1 2\n3 4\nB=\n5 6\n7 8", "19 22\n43 50"),
    ("A=\n-1 0 2\nB=\n3\n4\n5", "7"),
]


KRONECKER_FEW_SHOT = [
    ("A=\n1 2\nB=\n3 4", "3 4 6 8"),
    ("A=\n1\n2\nB=\n3 4", "3 4\n6 8"),
]


@dataclass
class MatrixMultiplicationProblem(Problem):
    """Prompt metadata and solver for matrix multiplication."""

    def __init__(self, few_shot_examples: list[tuple[str, str]] | None = None):
        super().__init__(
            name="Matrix Multiplication",
            delimiter=" ",
            regex_exp=MATRIX_REGEX,
            few_shot_examples=few_shot_examples or MATRIX_FEW_SHOT.copy(),
            solution=solve_matrix_multiplication,
        )

    def construct_prompt(self, test_input: str, few_shot_num: int = 8) -> str:
        return _matrix_prompt(
            task_instruction=(
                "You are given two input matrices: `A` and `B`. Every matrix element is provided row by row, "
                "separated by spaces. Matrix rows are separated by new lines. You have to calculate the matrix "
                "multiplication of `A` and `B` and print the matrix solution in the same format."
            ),
            few_shot_prompt=self.construct_few_shot_prompt(few_shot_num),
            test_input=test_input,
        )


@dataclass
class KroneckerProductProblem(Problem):
    """Prompt metadata and solver for Kronecker products."""

    def __init__(self, few_shot_examples: list[tuple[str, str]] | None = None):
        super().__init__(
            name="Kronecker Product",
            delimiter=" ",
            regex_exp=MATRIX_REGEX,
            few_shot_examples=few_shot_examples or KRONECKER_FEW_SHOT.copy(),
            solution=solve_kronecker_product,
        )

    def construct_prompt(self, test_input: str, few_shot_num: int = 8) -> str:
        return _matrix_prompt(
            task_instruction=(
                "You are given two input matrices: `A` and `B`. Calculate the Kronecker product `A ⊗ B` "
                "and print the matrix solution in the same row-by-row format."
            ),
            few_shot_prompt=self.construct_few_shot_prompt(few_shot_num),
            test_input=test_input,
        )


def _matrix_prompt(task_instruction: str, few_shot_prompt: str, test_input: str) -> str:
    return (
        f"{task_instruction}\n"
        f"{FENCE_INSTRUCTION}\n"
        f"{few_shot_prompt}"
        f"Please answer the following as demonstrated before:\n{test_input}\nSolution:\n"
    )
