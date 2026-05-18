from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable

from .base import FENCE, FENCE_INSTRUCTION, Problem, cycle_take

NUMERIC_REGEX = r"[+-]?\d+(?:\.\d+)?"
LIST_ROW_OR_DK_REGEX = r"(\[\s*[+-]?\d+(?:\.\d+)?(?:\s*,\s*[+-]?\d+(?:\.\d+)?)*\s*\])|(<DK>)|(<DK)|(DK>)|(DK)"
INTERVAL_OR_DK_REGEX = r"(\[\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*\])|(<DK>)|(<DK)|(DK>)|(DK)"


def two_sum(nums: list[int], target: int) -> list[int]:
    seen: dict[int, int] = {}
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return []


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, char_first in enumerate(first, start=1):
        current = [i]
        for j, char_second in enumerate(second, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_first != char_second),
                )
            )
        previous = current
    return previous[-1]


def rotate_matrix(matrix: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*matrix[::-1])]


def insert_interval(intervals: list[list[float]], new_interval: list[float]) -> list[list[float]]:
    merged: list[list[float]] = []
    start, end = new_interval
    inserted = False
    for current_start, current_end in intervals:
        if current_end < start:
            merged.append([current_start, current_end])
        elif end < current_start:
            if not inserted:
                merged.append([start, end])
                inserted = True
            merged.append([current_start, current_end])
        else:
            start = min(start, current_start)
            end = max(end, current_end)
    if not inserted:
        merged.append([start, end])
    return merged


def next_permutation(values: list[int]) -> list[int]:
    output = values.copy()
    pivot = len(output) - 2
    while pivot >= 0 and output[pivot] >= output[pivot + 1]:
        pivot -= 1
    if pivot >= 0:
        successor = len(output) - 1
        while output[successor] <= output[pivot]:
            successor -= 1
        output[pivot], output[successor] = output[successor], output[pivot]
    output[pivot + 1 :] = reversed(output[pivot + 1 :])
    return output


def reverse_words(text: str) -> str:
    return " ".join(word[::-1] for word in text.split())


def rotate_array(values: list[int], k: int) -> list[int]:
    if not values:
        return []
    shift = k % len(values)
    return values[-shift:] + values[:-shift] if shift else values.copy()


@dataclass
class CodeProblem(Problem):
    """Prompt metadata for deterministic code-output tasks."""

    description: str = ""
    textual_solution: str = ""
    variant: str = ""
    legacy_name: str = ""

    def __init__(
        self,
        name: str,
        description: str,
        solution: Callable,
        textual_solution: str | None = None,
        few_shot_examples: list[tuple[str, str]] | None = None,
        delimiter: str | None = " ",
        regex_exp: str | None = None,
        variant: str | None = None,
    ):
        self.description = description
        self.textual_solution = textwrap.dedent(textual_solution or inspect.getsource(solution)).strip()
        self.function_name = solution.__name__
        self.legacy_name = name
        self.variant = variant or name
        super().__init__(name="Code", delimiter=delimiter, regex_exp=regex_exp, few_shot_examples=few_shot_examples or [], solution=solution)

    def construct_prompt(self, test_input: str, few_shot_num: int = 8) -> str:
        few_shot_prompt = self.construct_few_shot_prompt(few_shot_num)
        return (
            "Given the following function and a call for the function with an input, generate the correct output "
            "for that input. Make sure to follow the structure as in the examples below.\n"
            f"{FENCE_INSTRUCTION}\n"
            f"Function Code:\n{self.textual_solution}\n"
            f"{few_shot_prompt}"
            f"Now, solve for the test Input.\nInput = {test_input}\n"
            f"output = {self.function_name}(*Input)\nprint('{FENCE}')\nprint(output)\nprint('{FENCE}')\nResponse:"
        )


class TwoSumProblem(CodeProblem):
    def __init__(self):
        examples = [
            ("([2, 7, 11, 15], 9)", str(two_sum([2, 7, 11, 15], 9))),
            ("([3, 2, 4], 6)", str(two_sum([3, 2, 4], 6))),
        ]
        super().__init__("TwoSumProblem", "Return indices of two numbers that sum to target.", two_sum, None, examples, delimiter=",", variant="Two Sum")


class EditDistanceProblem(CodeProblem):
    def __init__(self):
        examples = [("('kitten', 'sitting')", str(edit_distance("kitten", "sitting"))), ("('horse', 'ros')", str(edit_distance("horse", "ros")))]
        super().__init__(
            "EditDistanceProblem",
            "Return Levenshtein edit distance.",
            edit_distance,
            None,
            examples,
            delimiter=None,
            regex_exp=NUMERIC_REGEX,
            variant="Edit Distance",
        )


class RotateProblem(CodeProblem):
    def __init__(self):
        examples = [("([[1, 2], [3, 4]],)", str(rotate_matrix([[1, 2], [3, 4]]))), ("([[1]],)", str(rotate_matrix([[1]])))]
        super().__init__(
            "RotateProblem",
            "Rotate a square matrix clockwise.",
            rotate_matrix,
            None,
            examples,
            delimiter=None,
            regex_exp=LIST_ROW_OR_DK_REGEX,
            variant="Rotate Matrix",
        )


class InsertProblem(CodeProblem):
    def __init__(self):
        examples = [
            ("([[1, 3], [6, 9]], [2, 5])", str(insert_interval([[1, 3], [6, 9]], [2, 5]))),
            ("([[1, 2], [3, 5], [6, 7]], [4, 8])", str(insert_interval([[1, 2], [3, 5], [6, 7]], [4, 8]))),
        ]
        super().__init__(
            "InsertProblem",
            "Insert and merge an interval.",
            insert_interval,
            None,
            examples,
            delimiter=None,
            regex_exp=INTERVAL_OR_DK_REGEX,
            variant="Insert Interval",
        )


class NextPermutationProblem(CodeProblem):
    def __init__(self):
        examples = [("([1, 2, 3],)", str(next_permutation([1, 2, 3]))), ("([3, 2, 1],)", str(next_permutation([3, 2, 1])))]
        super().__init__(
            "NextPermutationProblem",
            "Return the next lexicographic permutation.",
            next_permutation,
            None,
            examples,
            delimiter=",",
            variant="Next Permutation",
        )


class ReverseWordsProblem(CodeProblem):
    def __init__(self):
        examples = [("('hello world',)", reverse_words("hello world")), ("('Let's take LeetCode contest',)", reverse_words("Let's take LeetCode contest"))]
        super().__init__("ReverseWordsProblem", "Reverse each word in a string.", reverse_words, None, examples, delimiter=" ", variant="Reverse Words")


class RotateArrayProblem(CodeProblem):
    def __init__(self):
        examples = [("([1, 2, 3, 4], 1)", str(rotate_array([1, 2, 3, 4], 1))), ("([1, 2, 3, 4], 2)", str(rotate_array([1, 2, 3, 4], 2)))]
        super().__init__("RotateArrayProblem", "Rotate an array to the right by k steps.", rotate_array, None, examples, delimiter=",", variant="Rotate Array")


def default_code_problems() -> list[CodeProblem]:
    """Return the built-in code task variants used by the dataset builder."""
    return [
        TwoSumProblem(),
        EditDistanceProblem(),
        RotateProblem(),
        InsertProblem(),
        NextPermutationProblem(),
        ReverseWordsProblem(),
        RotateArrayProblem(),
    ]


def code_rows(num_samples: int = 14) -> list[dict[str, str]]:
    """Return code task rows, cycling through built-in examples when needed."""
    rows: list[dict[str, str]] = []
    for problem in default_code_problems():
        for input_text, reference in problem.few_shot_examples:
            rows.append({"task": problem.name, "variant": problem.variant, "input": input_text, "reference": reference})
    return cycle_take(rows, num_samples)
