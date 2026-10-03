import pytest
from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import TurnParams

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def expected_outcome():
    return ConversationalGEval(
        name="Expected Outcome",
        criteria="Determine whether the assistant's answers across the conversation achieve the expected outcome with correct facts.",
        evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT, TurnParams.EXPECTED_OUTCOME],
        threshold=0.7,
        model=judge,
    )


@pytest.mark.parametrize("conv", cases("conversations"))
def test_expected_outcome_scripted(conv):
    assert_test(run_conversation(conv), [expected_outcome()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_expected_outcome_simulated(scenario):
    assert_test(simulate(scenario), [expected_outcome()])
