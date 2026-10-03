import pytest
from deepeval import assert_test
from deepeval.metrics import TurnContextualRelevancyMetric

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def turn_contextual_relevancy():
    return TurnContextualRelevancyMetric(threshold=0.4, model=judge)


@pytest.mark.parametrize("conv", cases("conversations"))
def test_turn_contextual_relevancy_scripted(conv):
    assert_test(run_conversation(conv), [turn_contextual_relevancy()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_turn_contextual_relevancy_simulated(scenario):
    assert_test(simulate(scenario), [turn_contextual_relevancy()])
