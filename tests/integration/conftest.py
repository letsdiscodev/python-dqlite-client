"""Integration test fixtures for dqlite-client.

These tests require a running dqlite cluster.
Start the test cluster with:
    cd ../dqlite-test-cluster && docker compose up -d
"""

import os

import pytest

# Skip integration tests if cluster not available
DQLITE_TEST_CLUSTER = os.environ.get("DQLITE_TEST_CLUSTER", "localhost:9001")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks tests as requiring dqlite cluster")


@pytest.fixture
def cluster_address() -> str:
    """Get the test cluster address."""
    return DQLITE_TEST_CLUSTER


@pytest.fixture
def cluster_addresses() -> list[str]:
    """Get all test cluster addresses."""
    return ["localhost:9001", "localhost:9002", "localhost:9003"]
