import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.helpers import generate, judge, single_turn_cases

pytestmark = pytest.mark.judged


def correctness():
    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether every fact in the expected output is present in the actual output.",
            "Penalize any fact in the actual output that contradicts the expected output.",
            "Do not penalize extra helpful details or different wording.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=judge,
    )


@pytest.mark.parametrize("golden", single_turn_cases())
def test_generator_correctness(golden):
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=generate(golden),
        expected_output=golden["expected_output"],
    )
    assert_test(test_case, [correctness()])
