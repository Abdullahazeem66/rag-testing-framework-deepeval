import pytest
from deepeval import assert_test
from deepeval.metrics import TurnContextualPrecisionMetric

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def turn_contextual_precision():
    return TurnContextualPrecisionMetric(threshold=0.6, model=judge)


@pytest.mark.parametrize("conv", cases("conversations"))
def test_turn_contextual_precision_scripted(conv):
    assert_test(run_conversation(conv), [turn_contextual_precision()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_turn_contextual_precision_simulated(scenario):
    assert_test(simulate(scenario), [turn_contextual_precision()])
