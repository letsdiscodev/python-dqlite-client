"""Integration test fixtures; require a running dqlite cluster
(``cd ../python-dqlite-dev/cluster && docker compose up -d``).

``pytest_plugins = ["dqlitetestlib.fixtures"]`` lives in the top-level conftest because
Pytest 8+ rejects ``pytest_plugins`` in non-top-level conftests.
"""

import os

import pytest

# Single bootstrap address; default matches the python-dqlite-dev cluster.
DQLITE_TEST_CLUSTER = os.environ.get("DQLITE_TEST_CLUSTER", "localhost:9001")
# Full node list (comma-separated); default matches the python-dqlite-dev 3-node cluster.
DQLITE_TEST_CLUSTER_NODES = os.environ.get(
    "DQLITE_TEST_CLUSTER_NODES",
    "localhost:9001,localhost:9002,localhost:9003",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as requiring dqlite cluster")


@pytest.fixture
def cluster_address() -> str:
    return DQLITE_TEST_CLUSTER


@pytest.fixture
def cluster_addresses() -> list[str]:
    return ["localhost:9001", "localhost:9002", "localhost:9003"]


@pytest.fixture
def cluster_node_addresses() -> list[str]:
    """Env-configurable node addresses for multi-node tests (prefer over hardcoded
    ``cluster_addresses``, kept only for backwards-compat with the shared fixture shape)."""
    return [a.strip() for a in DQLITE_TEST_CLUSTER_NODES.split(",") if a.strip()]
