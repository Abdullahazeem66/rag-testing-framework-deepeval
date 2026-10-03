import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from tests.helpers import judge, retrieve, single_turn_cases

pytestmark = pytest.mark.judged


def contextual_recall():
    return ContextualRecallMetric(threshold=0.7, model=judge)


@pytest.mark.parametrize("golden", single_turn_cases())
def test_contextual_recall(golden):
    test_case = LLMTestCase(
        input=golden["input"],
        expected_output=golden["expected_output"],
        retrieval_context=[c.text for c in retrieve(golden["input"])],
    )
    assert_test(test_case, [contextual_recall()])
