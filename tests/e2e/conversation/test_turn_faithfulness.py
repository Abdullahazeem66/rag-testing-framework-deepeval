import pytest
from deepeval import assert_test
from deepeval.metrics import TurnFaithfulnessMetric

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def turn_faithfulness():
    return TurnFaithfulnessMetric(threshold=0.8, model=judge)


@pytest.mark.parametrize("conv", cases("conversations"))
def test_turn_faithfulness_scripted(conv):
    assert_test(run_conversation(conv), [turn_faithfulness()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_turn_faithfulness_simulated(scenario):
    assert_test(simulate(scenario), [turn_faithfulness()])
