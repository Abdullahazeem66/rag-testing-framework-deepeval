import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.helpers import judge, retrieve, single_turn_cases

pytestmark = pytest.mark.judged


def contextual_relevancy():
    return ContextualRelevancyMetric(threshold=0.4, model=judge)


@pytest.mark.parametrize("golden", single_turn_cases())
def test_contextual_relevancy(golden):
    test_case = LLMTestCase(
        input=golden["input"],
        retrieval_context=[c.text for c in retrieve(golden["input"])],
    )
    assert_test(test_case, [contextual_relevancy()])
