import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.helpers import cases, generate, given_chunks, judge

pytestmark = pytest.mark.judged


def refusal():
    return GEval(
        name="Refusal",
        evaluation_steps=[
            "The retrieval context is related to the question but does not contain the answer.",
            "Check that the actual output clearly says the information is not available instead of inventing an answer.",
            "Reward pointing the user to Orbitly support; penalize any specific made-up facts, numbers, names or dates.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.7,
        model=judge,
    )


@pytest.mark.parametrize("golden", cases("unanswerable"))
def test_generator_refusal(golden):
    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=generate(golden),
        retrieval_context=[c.text for c in given_chunks(golden)],
    )
    assert_test(test_case, [refusal()])
