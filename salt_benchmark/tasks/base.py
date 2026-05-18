from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

FENCE = "********"
FENCE_INSTRUCTION = (
    "You can write anything you want to help yourself, but right before you write your answer, "
    "you should print '********' in a single line. At the end of your answer please place another "
    "'********' to signal the answer is finished. MAKE SURE YOU FENCE YOUR FINAL ANSWER WITH "
    "'********' AT THE BEGINNING AND THE END, OR ELSE, YOUR ANSWER WILL AUTOMATICALLY FAIL!"
)


@dataclass
class Problem:
    """Minimal task interface used by SALT dataset and prompt utilities."""

    name: str
    delimiter: str | None = " "
    regex_exp: str | None = None
    few_shot_examples: list[tuple[str, str]] = field(default_factory=list)
    solution: Callable[..., str] | None = None

    def construct_prompt(self, test_input, few_shot_num: int = 8) -> str:
        """Build a fenced generation prompt for one task input."""
        raise NotImplementedError

    def construct_few_shot_prompt(self, few_shot_num: int = 8) -> str:
        """Render up to ``few_shot_num`` examples using the task's fenced answer format."""
        examples = self.few_shot_examples[:few_shot_num]
        if not examples:
            return ""
        lines = [f"Here are {len(examples)} examples:"]
        for index, (example_input, example_output) in enumerate(examples, start=1):
            lines.extend(
                [
                    f"Example {index}:",
                    str(example_input),
                    "Answer:",
                    FENCE,
                    str(example_output),
                    FENCE,
                ]
            )
        return "\n".join(lines) + "\n"


def maybe_save_dataset(dataset, save: bool = False, save_dir: str | None = None):
    """Optionally save a HuggingFace Dataset to disk and return it."""
    if save:
        if not save_dir:
            raise ValueError("save_dir must be provided when save=True.")
        dataset.save_to_disk(save_dir)
    return dataset


def cycle_take(items: Sequence, num_items: int) -> list:
    """Return ``num_items`` values by cycling through a non-empty sequence."""
    if num_items < 0:
        raise ValueError("num_items must be non-negative.")
    if not items and num_items:
        raise ValueError("Cannot take samples from an empty sequence.")
    return [items[index % len(items)] for index in range(num_items)]
