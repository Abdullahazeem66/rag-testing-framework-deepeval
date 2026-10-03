import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from tests.helpers import ask, judge, single_turn_cases

pytestmark = pytest.mark.judged


def faithfulness():
    return FaithfulnessMetric(threshold=0.8, model=judge)


@pytest.mark.parametrize("golden", single_turn_cases())
def test_e2e_faithfulness(golden):
    resp = ask(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=resp.answer,
        retrieval_context=resp.retrieval_context,
    )
    assert_test(test_case, [faithfulness()])
