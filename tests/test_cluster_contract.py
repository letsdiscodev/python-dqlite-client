"""Leader discovery contract with a stubbed peer query."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from dqliteclient import ClusterClient, ClusterError, ClusterPolicyError, MemoryNodeStore
from dqliteclient.exceptions import DqliteConnectionError

Answers = dict[str, str | None | BaseException]


def stub(cluster: ClusterClient, answers: Answers, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    asked: list[str] = []

    async def query_leader(address: str, trust: bool) -> str | None:
        asked.append(address)
        answer = answers.get(address, DqliteConnectionError(f"Failed to connect to {address}"))
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(cluster, "_query_leader", query_leader)
    return asked


def make_cluster(*addresses: str, **kwargs: object) -> ClusterClient:
    return ClusterClient(MemoryNodeStore(list(addresses)), timeout=0.5, **kwargs)  # type: ignore[arg-type]


async def test_self_confirmed_leader_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1", "b:1")
    stub(cluster, {"a:1": "b:1", "b:1": "b:1"}, monkeypatch)
    assert await cluster.find_leader() == "b:1"


async def test_redirect_must_be_confirmed_by_target(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1")
    stub(cluster, {"a:1": "b:1", "b:1": "c:1"}, monkeypatch)
    with pytest.raises(ClusterError, match="did not confirm"):
        await cluster.find_leader()


async def test_all_peers_down_reports_every_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1", "b:1")
    stub(cluster, {}, monkeypatch)
    with pytest.raises(ClusterError, match="Could not find leader") as info:
        await cluster.find_leader()
    assert "a:1" in str(info.value) and "b:1" in str(info.value)


async def test_no_leader_known_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1")
    stub(cluster, {"a:1": None}, monkeypatch)
    with pytest.raises(ClusterError, match="no leader known"):
        await cluster.find_leader()


async def test_policy_rejection_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1", redirect_policy=lambda address: address != "evil:1")
    asked = stub(cluster, {"a:1": "evil:1", "evil:1": "evil:1"}, monkeypatch)
    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader()
    with pytest.raises(ClusterPolicyError):
        await cluster.connect()
    assert asked.count("a:1") == 2


async def test_malformed_redirect_target_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1", "b:1")
    stub(cluster, {"a:1": "bad\nhost:1", "b:1": "b:1"}, monkeypatch)
    assert await cluster.find_leader() == "b:1"


async def test_cached_leader_is_tried_first_then_forgotten(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1", "b:1")
    answers: Answers = {"a:1": "b:1", "b:1": "b:1"}
    asked = stub(cluster, answers, monkeypatch)
    assert await cluster.find_leader() == "b:1"
    asked.clear()
    assert await cluster.find_leader() == "b:1"
    assert asked == ["b:1"]
    answers["b:1"] = DqliteConnectionError("down")
    answers["a:1"] = "a:1"
    asked.clear()
    assert await cluster.find_leader() == "a:1"
    assert asked[0] == "b:1"


async def test_concurrent_callers_share_one_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1")
    calls = 0

    async def slow_query(address: str, trust: bool) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return address

    monkeypatch.setattr(cluster, "_query_leader", slow_query)
    results = await asyncio.gather(*(cluster.find_leader() for _ in range(5)))
    assert results == ["a:1"] * 5 and calls == 1


async def test_cancelled_caller_does_not_cancel_the_shared_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = make_cluster("a:1")

    async def slow_query(address: str, trust: bool) -> str:
        await asyncio.sleep(0.02)
        return address

    monkeypatch.setattr(cluster, "_query_leader", slow_query)
    first = asyncio.create_task(cluster.find_leader())
    second = asyncio.create_task(cluster.find_leader())
    await asyncio.sleep(0.005)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == "a:1"


async def test_connect_retries_transport_failures_then_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = make_cluster("a:1")
    attempts = 0

    async def find_leader(**kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        raise ClusterError("Could not find leader")

    monkeypatch.setattr(cluster, "find_leader", find_leader)
    with pytest.raises(ClusterError):
        await cluster.connect(max_attempts=2)
    assert attempts == 2


async def test_connect_validates_arguments() -> None:
    cluster = make_cluster("a:1")
    with pytest.raises(TypeError, match="max_attempts must be int"):
        await cluster.connect(max_attempts=True)
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        await cluster.connect(max_attempts=0)
    with pytest.raises(ValueError, match="max_elapsed_seconds"):
        await cluster.connect(max_elapsed_seconds=0)


Query = Callable[[str, bool], Awaitable[str | None]]


async def test_per_call_policy_overrides_instance_default(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = make_cluster("a:1", redirect_policy=lambda address: True)
    stub(cluster, {"a:1": "b:1", "b:1": "b:1"}, monkeypatch)
    assert await cluster.find_leader() == "b:1"
    with pytest.raises(ClusterPolicyError):
        await cluster.find_leader(policy=lambda address: address == "a:1")
