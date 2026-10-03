import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from tests.helpers import generate, given_chunks, judge, single_turn_cases

pytestmark = pytest.mark.judged


def faithfulness():
    return FaithfulnessMetric(threshold=0.8, model=judge)


@pytest.mark.parametrize("golden", single_turn_cases())
def test_generator_faithfulness(golden):
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=generate(golden),
        retrieval_context=[c.text for c in given_chunks(golden)],
    )
    assert_test(test_case, [faithfulness()])
