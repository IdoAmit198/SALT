from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from sympy import And, Equivalent, Implies, Not, Or, satisfiable, symbols
from sympy.logic.boolalg import BooleanFunction

from .base import FENCE_INSTRUCTION, Problem


def _symbol_sort_key(symbol) -> int:
    text = str(symbol)
    return int(text[1:]) if text.startswith("p") and text[1:].isdigit() else 0


def generate_random_formula(variables: tuple, rng: random.Random):
    """Generate one small random SymPy boolean formula over the provided variables."""
    operations = [And, Or, Not, Implies, Equivalent]
    formula = rng.choice(variables)
    for _ in range(rng.randint(1, 3)):
        operation = rng.choice(operations)
        if operation == Not:
            formula = operation(formula)
        else:
            formula = operation(formula, rng.choice(variables))
    return formula


def generate_formulas(num_variables: int, rng: random.Random, max_attempts: int = 500):
    """Generate a satisfiable list of boolean formulas and their variables."""
    variables = symbols(f"p1:{num_variables + 1}")
    formulas = []
    attempts = 0
    while len(formulas) < num_variables and attempts < max_attempts:
        attempts += 1
        formula = generate_random_formula(variables, rng)
        if satisfiable(And(*formulas, formula)):
            formulas.append(formula)
    if len(formulas) < num_variables:
        raise RuntimeError("Could not generate enough satisfiable formulas.")
    return formulas, variables


def assign_truth_values(variables: tuple, formulas: list, num_initial_assignments: int, rng: random.Random):
    """Sample initial truth assignments that are consistent with the formulas."""
    if num_initial_assignments < 1 or num_initial_assignments > len(variables):
        raise ValueError("num_initial_assignments must be between 1 and num_variables.")
    while True:
        selected = rng.sample(list(variables), num_initial_assignments)
        assignments = {variable: rng.choice([True, False]) for variable in selected}
        consistent = True
        for formula in formulas:
            formula_eval = formula.subs(assignments)
            if not isinstance(formula_eval, BooleanFunction) and not formula_eval:
                consistent = False
                break
        if consistent:
            return assignments


def get_variables_in_formulas(formulas: list) -> list:
    """Return variables appearing in formulas sorted by numeric suffix."""
    variables = set()
    for formula in formulas:
        variables.update(formula.free_symbols)
    return sorted(variables, key=_symbol_sort_key)


def deduce_truth_values(formulas: list, assignments: dict, all_variables: Iterable) -> dict | None:
    """Deduce variables whose truth value is forced by formulas and assignments."""
    variables_in_formulas = get_variables_in_formulas(formulas)
    assignment_constraints = [Equivalent(variable, value) for variable, value in assignments.items()]

    if not satisfiable(And(*formulas, *assignment_constraints)):
        return None

    deduced = assignments.copy()
    while True:
        changed = False
        for variable in variables_in_formulas:
            if variable in deduced:
                continue
            constraints = [Equivalent(var, val) for var, val in deduced.items()]
            sat_true = satisfiable(And(*formulas, *constraints, variable))
            sat_false = satisfiable(And(*formulas, *constraints, Not(variable)))
            if sat_true and not sat_false:
                deduced[variable] = True
                changed = True
            elif sat_false and not sat_true:
                deduced[variable] = False
                changed = True
        if not changed:
            break
    return {variable: deduced[variable] for variable in sorted(deduced, key=_symbol_sort_key)}


def formula_to_str(formula, notation: str = "prefix") -> str:
    """Render a SymPy boolean formula in SALT's prefix notation."""
    if notation != "prefix":
        raise ValueError("SALT currently supports prefix notation for the Logic task.")
    if isinstance(formula, And):
        return f"∧({', '.join(formula_to_str(argument, notation) for argument in formula.args)})"
    if isinstance(formula, Or):
        return f"∨({', '.join(formula_to_str(argument, notation) for argument in formula.args)})"
    if isinstance(formula, Not):
        return f"¬({formula_to_str(formula.args[0], notation)})"
    if isinstance(formula, Implies):
        return f"→({formula_to_str(formula.args[0], notation)}, {formula_to_str(formula.args[1], notation)})"
    if isinstance(formula, Equivalent):
        return f"↔({formula_to_str(formula.args[0], notation)}, {formula_to_str(formula.args[1], notation)})"
    return str(formula)


def create_logic_sample(
    num_variables: int = 8,
    num_initial_assignments: int = 3,
    rng: random.Random | None = None,
    notation: str = "prefix",
) -> tuple[str, str]:
    """Create one logic deduction input and its conclusive truth-value reference."""
    rng = rng or random.Random()
    while True:
        formulas, variables = generate_formulas(num_variables, rng)
        assignments = assign_truth_values(variables, formulas, num_initial_assignments, rng)
        conclusive_truth_values = deduce_truth_values(formulas, assignments, variables)
        if conclusive_truth_values is not None and len(assignments) < len(conclusive_truth_values):
            break

    formulas_text = "\n".join(formula_to_str(formula, notation) for formula in formulas)
    assignments_text = "".join(
        f"{variable}: {assignments[variable]}\n" for variable in sorted(assignments, key=_symbol_sort_key)
    )
    reference_text = "".join(
        f"{variable}: {value}\n" for variable, value in sorted(conclusive_truth_values.items(), key=lambda item: _symbol_sort_key(item[0]))
    )
    sample_input = f"{formulas_text}\n\nInitial Assignments:\n{assignments_text}"
    return sample_input, reference_text


LOGIC_FEW_SHOT = [
    (
        "∧(p3, ¬(p2))\n→(p4, p5)\n∨(p3, p7)\n\nInitial Assignments:\np3: True\n",
        "p2: False\np3: True\n",
    ),
    (
        "¬(p7)\n↔(p10, p12)\n∧(p4, ¬(p11))\n\nInitial Assignments:\np7: False\np10: True\np12: True\n",
        "p4: True\np7: False\np10: True\np11: False\np12: True\n",
    ),
]


@dataclass
class LogicProblem(Problem):
    """Prompt metadata for the First-Order Logic task."""

    def __init__(self, few_shot_examples: list[tuple[str, str]] | None = None):
        super().__init__(
            name="First-Order Logic",
            delimiter="\n",
            regex_exp=None,
            few_shot_examples=few_shot_examples or LOGIC_FEW_SHOT.copy(),
            solution=None,
        )

    def construct_prompt(self, test_input: str, few_shot_num: int = 8) -> str:
        few_shot_prompt = self.construct_few_shot_prompt(few_shot_num)
        return (
            "You are given logical formulas and initial truth assignments. Deduce every variable truth value "
            "that is forced by the formulas and print each conclusive assignment on its own line.\n"
            f"{FENCE_INSTRUCTION}\n"
            f"{few_shot_prompt}"
            f"Please answer the following as demonstrated before:\n{test_input}\nAnswer =\n"
        )
