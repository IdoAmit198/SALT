from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .base import FENCE, FENCE_INSTRUCTION, Problem

SUBJECTS = {
    "colors": ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"],
    "months": ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],
    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "numbers": ["1", "2", "3", "5", "8", "13", "21", "34"],
    "capital_letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
}

FILLER_WORDS = [
    "the",
    "archive",
    "recorded",
    "a",
    "quiet",
    "note",
    "before",
    "another",
    "entry",
    "appeared",
    "near",
    "the",
    "end",
]

NEEDLES_FEW_SHOT = [
    ("roses are red, violets are blue and watermelons are red and green", "red blue red green", "colors"),
    ("on monday the report moved from march to april before friday", "monday friday", "weekdays"),
]


def _tokenize_for_subject(text: str, subject: str) -> list[str]:
    if subject == "capital_letters":
        return [character for character in text if character.isupper()]
    if subject == "numbers":
        return re.findall(r"\d+(?:\.\d+)?", text)
    return re.findall(r"[A-Za-z]+", text.lower())


def find_occurrences(text: str, subject: str) -> str:
    """Return subject tokens found in text, preserving their order of occurrence."""
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject: {subject}")
    subject_set = set(SUBJECTS[subject])
    return " ".join(token for token in _tokenize_for_subject(text, subject) if token in subject_set)


def generate_needles_sample(rng: random.Random, min_occurrences: int = 2, max_occurrences: int = 6) -> tuple[str, str, str]:
    """Generate one text, subject name, and extraction reference."""
    subject = rng.choice(list(SUBJECTS.keys()))
    occurrence_count = rng.randint(min_occurrences, max_occurrences)
    needles = [rng.choice(SUBJECTS[subject]) for _ in range(occurrence_count)]
    tokens = FILLER_WORDS.copy()
    insert_positions = sorted(rng.sample(range(len(tokens) + occurrence_count), occurrence_count))
    needle_iter = iter(needles)
    output_tokens = []
    filler_iter = iter(tokens)
    for position in range(len(tokens) + occurrence_count):
        if position in insert_positions:
            output_tokens.append(next(needle_iter))
        else:
            output_tokens.append(next(filler_iter))
    text = " ".join(output_tokens)
    reference = find_occurrences(text, subject)
    return text, subject, reference


@dataclass
class NeedlesInHaystackProblem(Problem):
    """Prompt metadata and solver for the Words Collection task."""

    def __init__(self, few_shot_examples: list[tuple[str, str, str]] | None = None):
        self.needles_few_shot_examples = few_shot_examples or NEEDLES_FEW_SHOT.copy()
        super().__init__(
            name="Words Collection",
            delimiter=" ",
            regex_exp=None,
            few_shot_examples=[(f"List all the {subject} mentioned in the given text:\n{text}", output) for text, output, subject in self.needles_few_shot_examples],
            solution=find_occurrences,
        )

    def construct_prompt(self, test_input: dict | tuple[str, str], few_shot_num: int = 8) -> str:
        if isinstance(test_input, tuple):
            text, subject = test_input
        else:
            text, subject = test_input["text"], test_input["subject"]
        few_shot_prompt = self.construct_few_shot_prompt(few_shot_num)
        lower_case_instruction = " Make sure to use only lower case characters in your final answer." if subject not in {"capital_letters", "numbers"} else ""
        return (
            f"For the following text, please list all the {subject} mentioned in it according to their order in the story.\n"
            f"The delimiter between {subject} should only be a space ' '.{lower_case_instruction}\n"
            f"For example, if the text was 'roses are red, violets are blue and watermelons are red and green' then your answer should be 'red blue red green'.\n"
            f"{FENCE_INSTRUCTION}\n"
            f"{few_shot_prompt}"
            f"Please answer the following as demonstrated before:\nList all the {subject} mentioned in the given text:\n{text}\nAnswer: "
        )
