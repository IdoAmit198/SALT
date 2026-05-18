from __future__ import annotations

import random
from dataclasses import dataclass, field

from .base import FENCE, FENCE_INSTRUCTION, Problem

DNA_BASES = "ACTG"


def complementary_dna(seq: str, delimiter: str = " ") -> str:
    """Return the complementary DNA sequence, preserving codon delimiters."""
    if not seq:
        raise ValueError("Sequence is empty.")
    mapping = {"A": "T", "T": "A", "C": "G", "G": "C", delimiter: " "}
    output = []
    for base in seq:
        if base not in mapping:
            raise ValueError(f"Got a base outside ACTG: {base!r}")
        output.append(mapping[base])
    return delimiter.join("".join(output).split())


def generate_dna_sequence(rng: random.Random, min_codons: int = 1, max_codons: int = 30) -> str:
    """Generate a random codon-delimited DNA sequence."""
    if min_codons < 1 or max_codons < min_codons:
        raise ValueError("Expected 1 <= min_codons <= max_codons.")
    codon_count = rng.randint(min_codons, max_codons)
    codons = ["".join(rng.choice(DNA_BASES) for _ in range(3)) for _ in range(codon_count)]
    return " ".join(codons)


DEFAULT_DNA_FEW_SHOT = [
    ("AGA GCT AAG GCC", "TCT CGA TTC CGG"),
    ("CGG CTA AAG TCT GCG TAC TGG", "GCC GAT TTC AGA CGC ATG ACC"),
    ("GAT ATG GCC", "CTA TAC CGG"),
    ("TCT GCG ATG CCC ACC TCT", "AGA CGC TAC GGG TGG AGA"),
]


@dataclass
class DNAProblem(Problem):
    """Prompt metadata and solver for the DNA Translation task."""

    few_shot_examples: list[tuple[str, str]] = field(default_factory=lambda: DEFAULT_DNA_FEW_SHOT.copy())

    def __init__(self, few_shot_examples: list[tuple[str, str]] | None = None):
        super().__init__(
            name="DNA Translation",
            delimiter=" ",
            regex_exp=None,
            few_shot_examples=few_shot_examples or DEFAULT_DNA_FEW_SHOT.copy(),
            solution=complementary_dna,
        )

    def construct_prompt(self, test_input: str, few_shot_num: int = 8) -> str:
        few_shot_prompt = self.construct_few_shot_prompt(few_shot_num)
        return (
            "You are given a DNA sequence and required to print the complementary DNA sequence to form a double helix.\n"
            f"{FENCE_INSTRUCTION}\n"
            f"{few_shot_prompt}"
            f"Please answer the following as demonstrated before:\nGiven DNA sequence = {test_input}\nAnswer ="
        )
