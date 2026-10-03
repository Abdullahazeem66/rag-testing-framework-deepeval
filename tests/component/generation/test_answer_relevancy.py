import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.helpers import generate, judge, single_turn_cases

pytestmark = pytest.mark.judged


def answer_relevancy():
    return AnswerRelevancyMetric(threshold=0.7, model=judge)


@pytest.mark.parametrize("golden", single_turn_cases())
def test_generator_answer_relevancy(golden):
    test_case = LLMTestCase(input=golden["input"], actual_output=generate(golden))
    assert_test(test_case, [answer_relevancy()])
