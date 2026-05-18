from collections.abc import Mapping

from .code import (
    CodeProblem,
    EditDistanceProblem,
    InsertProblem,
    NextPermutationProblem,
    ReverseWordsProblem,
    RotateArrayProblem,
    RotateProblem,
    TwoSumProblem,
)
from .dna import DNAProblem, complementary_dna, generate_dna_sequence
from .logic import LogicProblem, create_logic_sample
from .matrix import (
    KroneckerProductProblem,
    MatrixMultiplicationProblem,
    generate_kronecker_product_problem,
    generate_matrix_multiplication_problem,
    solve_kronecker_product,
    solve_matrix_multiplication,
)
from .needles import NeedlesInHaystackProblem, find_occurrences, generate_needles_sample

PAPER_TASK_NAMES = (
    "Matrix Multiplication",
    "Kronecker Product",
    "Code",
    "First-Order Logic",
    "DNA Translation",
    "Words Collection",
)

PROBLEM_BY_TASK = {
    "DNA Translation": DNAProblem,
    "First-Order Logic": LogicProblem,
    "Matrix Multiplication": MatrixMultiplicationProblem,
    "Kronecker Product": KroneckerProductProblem,
    "Words Collection": NeedlesInHaystackProblem,
}

CODE_PROBLEM_BY_VARIANT = {
    "Two Sum": TwoSumProblem,
    "Edit Distance": EditDistanceProblem,
    "Rotate Matrix": RotateProblem,
    "Insert Interval": InsertProblem,
    "Next Permutation": NextPermutationProblem,
    "Reverse Words": ReverseWordsProblem,
    "Rotate Array": RotateArrayProblem,
}

LEGACY_PROBLEM_BY_TASK = {
    "DNAProblem": DNAProblem,
    "LogicProblem": LogicProblem,
    "MatrixMultiplicationProblem": MatrixMultiplicationProblem,
    "KroneckerProductProblem": KroneckerProductProblem,
    "NeedlesInHaystackProblem": NeedlesInHaystackProblem,
    "TwoSumProblem": TwoSumProblem,
    "EditDistanceProblem": EditDistanceProblem,
    "RotateProblem": RotateProblem,
    "InsertProblem": InsertProblem,
    "NextPermutationProblem": NextPermutationProblem,
    "ReverseWordsProblem": ReverseWordsProblem,
    "RotateArrayProblem": RotateArrayProblem,
}


def problem_from_task(row_or_task: Mapping[str, object] | str):
    """Instantiate the task problem referenced by a dataset row or task name."""
    task_name = row_or_task["task"] if isinstance(row_or_task, Mapping) else row_or_task
    if not isinstance(task_name, str):
        raise TypeError("Expected a task name string or a row with a string 'task' field.")
    if task_name == "Code":
        if not isinstance(row_or_task, Mapping):
            raise ValueError("Code task rows require a 'variant' field; pass a dataset row or a specific code variant.")
        variant = row_or_task.get("variant")
        if not isinstance(variant, str):
            raise ValueError("Code task rows require a string 'variant' field.")
        try:
            return CODE_PROBLEM_BY_VARIANT[variant]()
        except KeyError as exc:
            known_variants = ", ".join(sorted(CODE_PROBLEM_BY_VARIANT))
            raise ValueError(f"Unknown Code variant {variant!r}. Known variants: {known_variants}") from exc
    try:
        problem_type = PROBLEM_BY_TASK[task_name]
    except KeyError as exc:
        problem_type = LEGACY_PROBLEM_BY_TASK.get(task_name)
        if problem_type is not None:
            return problem_type()
        known_tasks = ", ".join(PAPER_TASK_NAMES)
        raise ValueError(f"Unknown SALT task {task_name!r}. Known tasks: {known_tasks}") from exc
    return problem_type()

__all__ = [
    "CodeProblem",
    "DNAProblem",
    "EditDistanceProblem",
    "InsertProblem",
    "KroneckerProductProblem",
    "LogicProblem",
    "MatrixMultiplicationProblem",
    "NeedlesInHaystackProblem",
    "NextPermutationProblem",
    "PAPER_TASK_NAMES",
    "PROBLEM_BY_TASK",
    "ReverseWordsProblem",
    "RotateArrayProblem",
    "RotateProblem",
    "TwoSumProblem",
    "complementary_dna",
    "create_logic_sample",
    "find_occurrences",
    "generate_dna_sequence",
    "generate_kronecker_product_problem",
    "generate_matrix_multiplication_problem",
    "generate_needles_sample",
    "problem_from_task",
    "solve_kronecker_product",
    "solve_matrix_multiplication",
]
