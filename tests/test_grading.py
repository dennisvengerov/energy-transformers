import time

import pytest
from common import format_prompt
from grading import ExecResult, is_correct, parse_gold, run_program, run_programs, self_test

PROMPT = format_prompt("Tom has 30 apples and buys 10 more. How many apples does he have?")


def program(body):
    return PROMPT + "".join(f"    {line}\n" for line in body.split("\n"))


@pytest.mark.parametrize("body, gold, status, correct", [
    ("result = 30 + 10\nreturn result", 40, "ok", True),
    ("result = 80 / 2\nreturn result", 40, "ok", True),  # 40.0 vs 40
    ("return 0.1 * 3 * 1000 / 7.5", 40, "ok", True),  # float rounding within tolerance
    ("return 40.5", 40, "ok", False),
    ("import numpy as np\nreturn np.float64(40)", 40, "ok", True),
    ("print(f'<<<{40}>>>')\nprint('{\"status\": \"ok\", \"value\": 1}')\nreturn 40", 40, "ok", True),  # prints ignored
    ("return '40'", 40, "non_numeric", False),
    ("return True", 1, "non_numeric", False),
    ("return None", 40, "non_numeric", False),
    ("return float('nan')", 40, "non_numeric", False),
    ("return 1 / 0", 40, "error", False),
    ("return undefined_name", 40, "error", False),
    ("result = (30 +\nreturn result", 40, "error", False),  # syntax error
    ("with open('out.txt', 'w') as f:\n    f.write('x' * 100)\nreturn 40", 40, "error", False),  # file writes blocked
])
def test_grading(body, gold, status, correct):
    result = run_program(program(body))
    assert result.status == status, result
    assert is_correct(result, gold) == correct


def test_self_test_passes():
    self_test()


def test_missing_function_is_an_error():
    assert run_program("x = 40\n").status == "error"


def test_infinite_loop_times_out():
    t0 = time.time()
    result = run_program(program("while True:\n    pass"), timeout=1.0)
    assert result.status == "timeout" and not is_correct(result, 40)
    assert time.time() - t0 < 5


def test_run_programs_keeps_order():
    results = run_programs([program(f"return {i}") for i in range(12)], workers=4)
    assert [r.value for r in results] == list(range(12))


def test_parse_gold():
    assert parse_gold("He has 1,234 now.\n#### 1,234") == 1234.0
    assert parse_gold("#### -5") == -5.0
    assert parse_gold("#### 2.5") == 2.5


def test_only_ok_results_can_be_correct():
    assert not is_correct(ExecResult("error", value=40.0), 40)
    assert is_correct(ExecResult("ok", value=39.99999999), 40)
