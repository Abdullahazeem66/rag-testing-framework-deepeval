import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from tests.helpers import generate, ground_truth, judge, single_turn_cases

pytestmark = pytest.mark.judged


def hallucination():
    return HallucinationMetric(threshold=0.3, model=judge)


@pytest.mark.parametrize("golden", single_turn_cases())
def test_generator_hallucination(golden):
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=generate(golden),
        context=ground_truth(golden),
    )
    assert_test(test_case, [hallucination()])
