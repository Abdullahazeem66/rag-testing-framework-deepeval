import pytest
from deepeval import assert_test
from deepeval.metrics import RoleAdherenceMetric

from tests.helpers import cases, judge, run_conversation, scenario_cases, simulate

pytestmark = pytest.mark.judged


def role_adherence():
    return RoleAdherenceMetric(threshold=0.7, model=judge)


@pytest.mark.parametrize("conv", cases("conversations"))
def test_role_adherence_scripted(conv):
    assert_test(run_conversation(conv), [role_adherence()])


@pytest.mark.slow
@pytest.mark.parametrize("scenario", scenario_cases())
def test_role_adherence_simulated(scenario):
    assert_test(simulate(scenario), [role_adherence()])
