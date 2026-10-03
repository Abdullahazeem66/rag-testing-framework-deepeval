import pytest
from deepeval import assert_test
from deepeval.metrics import TurnRelevancyMetric

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def turn_relevancy():
    return TurnRelevancyMetric(threshold=0.7, model=judge)


@pytest.mark.parametrize("conv", cases("conversations"))
def test_turn_relevancy_scripted(conv):
    assert_test(run_conversation(conv), [turn_relevancy()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_turn_relevancy_simulated(scenario):
    assert_test(simulate(scenario), [turn_relevancy()])
