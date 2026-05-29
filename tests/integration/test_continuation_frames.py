"""Integration test for multi-frame rows continuation: query a result set large enough that
the server emits continuation frames (``has_more=True``), exercising the full
server -> wire -> ``_drain_continuations`` -> client reassembly path end-to-end."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dqliteclient import connect


@pytest.mark.integration
class TestContinuationFrames:
    async def test_large_result_set_crosses_continuation_boundary(
        self, cluster_address: str
    ) -> None:
        """Select enough rows to cross the continuation boundary; verify exact reassembly.

        ~60 bytes/row vs a tens-of-KiB per-frame buffer, so a few thousand rows guarantees it.
        """
        N_ROWS = 5000
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_continuation")
            await conn.execute("CREATE TABLE test_continuation (id INTEGER PRIMARY KEY, val TEXT)")

            async with conn.transaction():
                # SQLite caps parameters per statement at 999 by default; chunk to stay under.
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

            mid = N_ROWS // 2
            assert rows[mid][0] == mid

    async def test_continuation_boundary_actually_crossed(self, cluster_address: str) -> None:
        """Instrument ``_drain_continuations`` and assert ``frames > 1`` — a single-frame
        result set would give a false sense of coverage."""
        from dqliteclient.protocol import DqliteProtocol

        N_ROWS = 5000
        frames_seen: list[int] = []

        async def _spy(self, initial, deadline):
            # Structural copy of the real method plus a frames counter.
            response = initial
            all_rows = list(initial.rows)
            all_row_types: list[list[int]] = [[int(t) for t in rt] for rt in initial.row_types]
            frames = 1
            while response.has_more:
                next_response = await self._read_continuation(deadline=deadline)
                frames += 1
                if not next_response.rows and next_response.has_more:
                    break  # defer to original's error handling
                all_rows.extend(next_response.rows)
                all_row_types.extend([int(t) for t in rt] for rt in next_response.row_types)
                response = next_response
            frames_seen.append(frames)
            return all_rows, all_row_types

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
