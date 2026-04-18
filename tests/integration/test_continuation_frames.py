"""Integration test for multi-frame rows continuation (ISSUE-66).

The dqlite server batches query results into frames that fit in its
per-response buffer. For result sets larger than one frame's worth of
rows, the server sends an initial ``ROWS`` response with
``has_more=True`` followed by one or more continuation frames ending
with a final frame that sets ``has_more=False``.

Previously only a Python-round-trip test exercised this path
(``test_decode_continuation_roundtrip``). This integration test
queries a real dqlite server for a result set large enough to cross
the continuation boundary, exercising the full server→wire→
``_drain_continuations``→client path end-to-end.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient import connect


@pytest.mark.integration
class TestContinuationFrames:
    async def test_large_result_set_crosses_continuation_boundary(
        self, cluster_address: str
    ) -> None:
        """Select a result set large enough that the server emits at
        least one continuation frame. Verify row reassembly is exact.

        Row size: 8-byte INTEGER + ~50-byte TEXT per row ≈ 60 bytes
        plus tuple framing. The server's per-frame buffer is on the
        order of tens of KiB, so a few thousand rows guarantees at
        least one continuation.
        """
        N_ROWS = 5000
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_continuation")
            await conn.execute("CREATE TABLE test_continuation (id INTEGER PRIMARY KEY, val TEXT)")

            # Batch inserts into a single transaction for speed.
            async with conn.transaction():
                # SQLite has a default compile-time limit of 999
                # parameters per statement; chunk to stay under.
                batch = 500
                for start in range(0, N_ROWS, batch):
                    values = []
                    params: list[object] = []
                    for i in range(start, min(start + batch, N_ROWS)):
                        values.append("(?, ?)")
                        params.extend([i, f"row-{i:06d}-padding-padding"])
                    await conn.execute(
                        "INSERT INTO test_continuation (id, val) VALUES " + ",".join(values),
                        params,
                    )

            rows = await conn.fetchall("SELECT id, val FROM test_continuation ORDER BY id")
            assert len(rows) == N_ROWS
            assert rows[0][0] == 0
            assert rows[0][1] == "row-000000-padding-padding"
            assert rows[-1][0] == N_ROWS - 1
            assert rows[-1][1] == f"row-{N_ROWS - 1:06d}-padding-padding"

            # Spot-check a middle row.
            mid = N_ROWS // 2
            assert rows[mid][0] == mid

    async def test_continuation_boundary_actually_crossed(self, cluster_address: str) -> None:
        """Confirm the server actually emitted at least one
        continuation frame — a smaller result set that still fits in
        one frame would give a false sense of coverage. Instrument
        ``_drain_continuations`` and assert ``frames > 1``.
        """
        from dqliteclient.protocol import DqliteProtocol

        N_ROWS = 5000
        frames_seen: list[int] = []

        async def _spy(self, initial, deadline):  # type: ignore[no-untyped-def]
            # Copied structurally from the real method but with a
            # frames counter escape hatch. Keeps the original on the
            # class intact for other tests.
            response = initial
            all_rows = list(initial.rows)
            frames = 1
            while response.has_more:
                next_response = await self._read_continuation(deadline=deadline)
                frames += 1
                if not next_response.rows and next_response.has_more:
                    break  # defer to original's error handling
                all_rows.extend(next_response.rows)
                response = next_response
            frames_seen.append(frames)
            return all_rows

        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_cont_spy")
            await conn.execute("CREATE TABLE test_cont_spy (id INTEGER PRIMARY KEY, val TEXT)")
            async with conn.transaction():
                batch = 500
                for start in range(0, N_ROWS, batch):
                    values = []
                    params: list[object] = []
                    for i in range(start, min(start + batch, N_ROWS)):
                        values.append("(?, ?)")
                        params.extend([i, f"row-{i:06d}-pad"])
                    await conn.execute(
                        "INSERT INTO test_cont_spy (id, val) VALUES " + ",".join(values),
                        params,
                    )

            with patch.object(DqliteProtocol, "_drain_continuations", _spy):
                rows = await conn.fetchall("SELECT id, val FROM test_cont_spy")

            assert len(rows) == N_ROWS
            assert frames_seen, "spy never recorded a drain"
            assert any(f > 1 for f in frames_seen), (
                f"Server emitted only single-frame responses for {N_ROWS} rows; "
                f"frames_seen={frames_seen}. Test setup must produce a result "
                f"set large enough to cross the continuation boundary — raise "
                f"N_ROWS until frames > 1."
            )
