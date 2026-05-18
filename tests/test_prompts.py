from salt_benchmark.tasks import (
    DNAProblem,
    LogicProblem,
    MatrixMultiplicationProblem,
    NeedlesInHaystackProblem,
    PROBLEM_BY_TASK,
    TwoSumProblem,
    problem_from_task,
)


def test_dna_regular_prompt_contains_fences_and_few_shot():
    prompt = DNAProblem().construct_prompt("AGA GCT", few_shot_num=2)
    assert "********" in prompt
    assert prompt.count("Example ") == 2
    assert "Given DNA sequence = AGA GCT" in prompt


def test_logic_prompt_uses_newline_delimiter_metadata():
    problem = LogicProblem()
    prompt = problem.construct_prompt("p1\n\nInitial Assignments:\np1: True\n", few_shot_num=1)
    assert problem.delimiter == "\n"
    assert "Deduce every variable truth value" in prompt
    assert "Initial Assignments" in prompt


def test_matrix_prompt_preserves_regex_metadata():
    problem = MatrixMultiplicationProblem()
    prompt = problem.construct_prompt("A=\n1 2\nB=\n3\n4", few_shot_num=1)
    assert problem.regex_exp is not None
    assert "matrix multiplication" in prompt.lower()
    assert "Solution:" in prompt


def test_needles_prompt_accepts_dict_input():
    problem = NeedlesInHaystackProblem()
    prompt = problem.construct_prompt({"text": "red blue red", "subject": "colors"}, few_shot_num=1)
    assert "List all the colors" in prompt
    assert "red blue red" in prompt


def test_code_prompt_contains_full_function_body():
    prompt = TwoSumProblem().construct_prompt("([3, 2, 4], 6)", few_shot_num=1)
    assert "def two_sum(nums: list[int], target: int) -> list[int]:" in prompt
    assert "for index, value in enumerate(nums):" in prompt
    assert "def two_sum(nums, target): ..." not in prompt


def test_problem_from_task_instantiates_dataset_task_rows():
    row = {"task": "Matrix Multiplication", "input": "A=", "reference": ""}
    assert isinstance(problem_from_task(row), MatrixMultiplicationProblem)
    assert isinstance(problem_from_task({"task": "Code", "variant": "Two Sum", "input": "", "reference": ""}), TwoSumProblem)
    assert isinstance(problem_from_task("TwoSumProblem"), TwoSumProblem)
    assert "Words Collection" in PROBLEM_BY_TASK
