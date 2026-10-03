import pytest

from tests.helpers import cases, retrieve

pytestmark = pytest.mark.deterministic


@pytest.mark.parametrize("golden", cases("single_turn"))
def test_gold_chunks_retrieved(golden):
    retrieved = [c.chunk_id for c in retrieve(golden["input"])]
    missing = set(golden["gold_chunks"]) - set(retrieved)
    assert not missing, f"missing {missing}, retrieved {retrieved}"


@pytest.mark.parametrize("golden", cases("single_turn"))
def test_first_gold_chunk_ranked_high(golden):
    retrieved = [c.chunk_id for c in retrieve(golden["input"])]
    rank = next((i for i, cid in enumerate(retrieved, 1) if cid in golden["gold_chunks"]), None)
    assert rank is not None and rank <= 2, f"first gold chunk at rank {rank}, retrieved {retrieved}"
