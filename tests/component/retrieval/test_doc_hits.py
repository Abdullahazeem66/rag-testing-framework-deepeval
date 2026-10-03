import pytest

from tests.helpers import retrieve, single_turn_cases

pytestmark = pytest.mark.deterministic


@pytest.mark.parametrize("golden", single_turn_cases())
def test_expected_docs_retrieved(golden):
    retrieved = [c.doc_id for c in retrieve(golden["input"])]
    missing = set(golden["expected_docs"]) - set(retrieved)
    assert not missing, f"missing {missing}, retrieved {retrieved}"


@pytest.mark.parametrize("golden", single_turn_cases())
def test_first_relevant_doc_ranked_high(golden):
    retrieved = [c.doc_id for c in retrieve(golden["input"])]
    rank = next((i for i, doc in enumerate(retrieved, 1) if doc in golden["expected_docs"]), None)
    assert rank is not None and rank <= 2, f"first relevant doc at rank {rank}, retrieved {retrieved}"
