import pytest
from deepeval import assert_test
from deepeval.metrics import ConversationCompletenessMetric

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def conversation_completeness():
    return ConversationCompletenessMetric(threshold=0.7, model=judge)


@pytest.mark.parametrize("conv", cases("conversations"))
def test_conversation_completeness_scripted(conv):
    assert_test(run_conversation(conv), [conversation_completeness()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_conversation_completeness_simulated(scenario):
    assert_test(simulate(scenario), [conversation_completeness()])
