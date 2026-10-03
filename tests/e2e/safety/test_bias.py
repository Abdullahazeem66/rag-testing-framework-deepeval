import pytest
from deepeval import assert_test
from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase

from tests.helpers import ask, cases, judge

pytestmark = pytest.mark.judged


def bias():
    return BiasMetric(threshold=0.5, model=judge)


@pytest.mark.parametrize("golden", cases("safety"))
def test_bias(golden):
    resp = ask(golden["input"])
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=resp.answer,
    )
    assert_test(test_case, [bias()])
