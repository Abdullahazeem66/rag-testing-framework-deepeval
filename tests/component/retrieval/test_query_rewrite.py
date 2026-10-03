import pytest

from tests.helpers import assistant, cases

pytestmark = pytest.mark.deterministic


@pytest.mark.parametrize("case", cases("query_rewrite"))
def test_follow_up_rewritten_to_standalone_query(case):
    query, _ = assistant().rewrite_query(case["message"], case["history"])
    missing = [term for term in case["must_contain"] if term not in query.lower()]
    assert not missing, f"rewritten query {query!r} is missing {missing}"
