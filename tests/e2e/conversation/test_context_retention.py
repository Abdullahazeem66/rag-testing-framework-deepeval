import pytest
from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import TurnParams

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def context_retention():
    return ConversationalGEval(
        name="Context Retention",
        criteria="Determine whether the assistant correctly remembers details the user gave in earlier turns and correctly resolves follow-up references such as 'it', 'that' or 'then' without asking the user to repeat themselves.",
        evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
        threshold=0.7,
        model=judge,
    )


@pytest.mark.parametrize("conv", cases("conversations"))
def test_context_retention_scripted(conv):
    assert_test(run_conversation(conv), [context_retention()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_context_retention_simulated(scenario):
    assert_test(simulate(scenario), [context_retention()])
