"""Pin: ``_estimate_request_body_size`` counts the ``sql`` text
field, not just ``params``, when deciding whether to offload the
encode to a worker thread.

The prior estimator walked ``request.params`` only. Two request
shapes therefore defeated the offload entirely:

* ``PrepareRequest`` has a ``sql`` field but NO ``params`` — a
  multi-MiB ``prepare(sql)`` always estimated to 0 and encoded
  in-loop.
* ``ExecSqlRequest`` / ``QuerySqlRequest`` carry both ``sql`` and
  ``params``; an inline-literal bulk statement (multi-MiB SQL,
  empty/small params — the shape SQLAlchemy ``insertmanyvalues``
  and large literal ``VALUES`` / ``IN (...)`` expansions produce)
  always estimated below the gate and encoded in-loop.

``sql`` is the single largest TEXT field on the request side
(capped at ~64 MiB), and ``encode_text`` does a UTF-8 transcode
plus a full-size copy — the exact loop-freeze the param-offload
was built to prevent, on a field it never measured.

The fix adds ``len(sql) * 4`` (the same pessimistic UTF-8
upper-bound already used for ``str`` params) to the estimate
when the request carries a str ``sql`` field.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dqliteclient import protocol as protocol_mod
from dqliteclient.protocol import DqliteProtocol
from dqlitewire.messages import requests as wire_requests


def _make_protocol_with_mock_writer() -> DqliteProtocol:
    proto = DqliteProtocol.__new__(DqliteProtocol)
    proto._writer = MagicMock()
    proto._writer.write = MagicMock()
    proto._writer.drain = AsyncMock()
    proto._timeout = 5.0
    from dqlitewire import MessageEncoder

    proto._encoder = MessageEncoder()
    proto._client_id = 1
    return proto


def test_estimate_counts_prepare_request_sql() -> None:
    """A ``PrepareRequest`` (sql, no params) with a multi-MiB SQL
    must estimate above the offload threshold. Pre-fix it scored 0.
    """
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    big_sql = "x" * (threshold)  # *4 in the estimate → comfortably over
    request = wire_requests.PrepareRequest(db_id=1, sql=big_sql)
    estimate = protocol_mod._estimate_request_body_size(request)
    assert estimate >= threshold, (
        f"PrepareRequest with {len(big_sql)}-char SQL estimated to "
        f"{estimate}, below the {threshold}-byte gate; the estimator "
        f"must count the ``sql`` field, not just ``params``."
    )


def test_estimate_counts_exec_sql_inline_literal() -> None:
    """An ``ExecSqlRequest`` with a multi-MiB SQL and empty params
    (inline-literal bulk insert) must estimate above the gate.
    """
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    big_sql = "INSERT INTO t VALUES " + "(1)," * (threshold // 4)
    request = wire_requests.ExecSqlRequest(db_id=1, sql=big_sql, params=[])
    estimate = protocol_mod._estimate_request_body_size(request)
    assert estimate >= threshold, (
        f"ExecSqlRequest with a {len(big_sql)}-char inline-literal SQL "
        f"estimated to {estimate}, below the {threshold}-byte gate."
    )


def test_estimate_small_sql_stays_below_threshold() -> None:
    """A normal small SQL must NOT push the estimate over the gate
    (no false-positive thread-hop on the common case).
    """
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD
    request = wire_requests.PrepareRequest(db_id=1, sql="SELECT 1")
    estimate = protocol_mod._estimate_request_body_size(request)
    assert estimate < threshold, (
        f"small PrepareRequest estimated to {estimate}; must stay below "
        f"the {threshold}-byte gate so heartbeat-class statements do not "
        f"pay the thread-hop cost."
    )


@pytest.mark.asyncio
async def test_prepare_request_large_sql_dispatches_via_to_thread() -> None:
    """End-to-end: a ``PrepareRequest`` with a multi-MiB SQL routed
    through ``_send_request`` must encode on a worker thread.
    """
    proto = _make_protocol_with_mock_writer()
    threshold = protocol_mod._ENCODE_OFFLOAD_THRESHOLD

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    big_sql = "x" * threshold
    request = wire_requests.PrepareRequest(db_id=1, sql=big_sql)

    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(request)

    assert len(to_thread_calls) == 1, (
        f"large-SQL PrepareRequest must offload the encode to "
        f"asyncio.to_thread; got {len(to_thread_calls)} hops."
    )
    assert proto._writer.write.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_small_prepare_request_stays_in_loop() -> None:
    """A small ``PrepareRequest`` must NOT offload."""
    proto = _make_protocol_with_mock_writer()

    to_thread_calls: list[Any] = []
    real_to_thread = asyncio.to_thread

    async def _tracking_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    request = wire_requests.PrepareRequest(db_id=1, sql="SELECT 1")
    with patch.object(asyncio, "to_thread", _tracking_to_thread):
        await proto._send_request(request)

    assert to_thread_calls == [], (
        f"small PrepareRequest unexpectedly offloaded: {to_thread_calls!r}"
    )
