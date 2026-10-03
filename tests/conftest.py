import pytest

from rag_app.ingest import build_index


def pytest_collection_modifyitems(items):
    for item in items:
        parts = item.path.parts
        if "component" in parts:
            item.add_marker(pytest.mark.component)
        elif "e2e" in parts:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session", autouse=True)
def index():
    build_index()
