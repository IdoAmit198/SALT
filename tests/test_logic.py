import random

from sympy import And, Equivalent, Implies, Not, Or, satisfiable, symbols

from salt_benchmark.tasks.logic import (
    LOGIC_FEW_SHOT,
    assign_truth_values,
    deduce_truth_values,
    generate_formulas,
    get_variables_in_formulas,
)


def test_logic_first_few_shot_lists_only_forced_assignments():
    p2, p3, p4, p5, p7 = symbols("p2 p3 p4 p5 p7")
    formulas = [And(p3, Not(p2)), Implies(p4, p5), Or(p3, p7)]
    assignments = {p3: True}

    assert deduce_truth_values(formulas, assignments, (p2, p3, p4, p5, p7)) == {p2: False, p3: True}
    assert LOGIC_FEW_SHOT[0][1] == "p2: False\np3: True\n"

    constraints = [Equivalent(variable, value) for variable, value in assignments.items()]
    for variable in (p5, p7):
        assert satisfiable(And(*formulas, *constraints, variable))
        assert satisfiable(And(*formulas, *constraints, Not(variable)))


def test_generated_logic_samples_include_all_and_only_forced_values():
    for seed in range(10):
        rng = random.Random(seed)
        formulas, variables = generate_formulas(6, rng)
        assignments = assign_truth_values(variables, formulas, 2, rng)
        deduced = deduce_truth_values(formulas, assignments, variables)
        assert deduced is not None

        constraints = [Equivalent(variable, value) for variable, value in assignments.items()]
        assert satisfiable(And(*formulas, *constraints))
        for variable in get_variables_in_formulas(formulas):
            sat_true = bool(satisfiable(And(*formulas, *constraints, variable)))
            sat_false = bool(satisfiable(And(*formulas, *constraints, Not(variable))))
            if variable in deduced:
                assert sat_true != sat_false
                assert deduced[variable] is sat_true
            else:
                assert sat_true and sat_false