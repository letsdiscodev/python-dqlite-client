"""Integration tests for single node operations. Requires a running cluster."""

import datetime

import pytest

from dqliteclient import connect


@pytest.mark.integration
class TestSingleNodeOperations:
    async def test_connect_and_query(self, cluster_address: str) -> None:
        conn = await connect(cluster_address)
        try:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
        finally:
            await conn.close()

    async def test_create_table_and_insert(self, cluster_address: str) -> None:
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_single")
            await conn.execute("CREATE TABLE test_single (id INTEGER PRIMARY KEY, name TEXT)")
            await conn.execute("INSERT INTO test_single (name) VALUES (?)", ["test"])

            rows = await conn.fetch("SELECT * FROM test_single WHERE name = ?", ["test"])
            assert len(rows) == 1
            assert rows[0]["name"] == "test"

    async def test_transaction_commit(self, cluster_address: str) -> None:
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_tx")
            await conn.execute("CREATE TABLE test_tx (id INTEGER PRIMARY KEY, val TEXT)")

            async with conn.transaction():
                await conn.execute("INSERT INTO test_tx (val) VALUES (?)", ["committed"])

            rows = await conn.fetch("SELECT * FROM test_tx WHERE val = ?", ["committed"])
            assert len(rows) == 1

    async def test_transaction_rollback(self, cluster_address: str) -> None:
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_rollback")
            await conn.execute("CREATE TABLE test_rollback (id INTEGER PRIMARY KEY, val TEXT)")

            try:
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO test_rollback (val) VALUES (?)", ["rollback_test"]
                    )
                    raise ValueError("Force rollback")
            except ValueError:
                pass

            rows = await conn.fetch("SELECT * FROM test_rollback WHERE val = ?", ["rollback_test"])
            assert len(rows) == 0

    async def test_unicode_text(self, cluster_address: str) -> None:
        """Unicode text: emojis, CJK, RTL, combining chars."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_unicode")
            await conn.execute("CREATE TABLE test_unicode (id INTEGER PRIMARY KEY, val TEXT)")

            unicode_values = [
                "Hello 🎉 World",
                "🎉🎊🎁🎂",
                "中文测试",
                "日本語テスト",
                "한국어 테스트",
                "العربية",
                "עברית",
                "Hello 世界 🌍",
                "café résumé naïve",
                "日本語" * 50,
            ]

            for val in unicode_values:
                await conn.execute("INSERT INTO test_unicode (val) VALUES (?)", [val])

                rows = await conn.fetch("SELECT val FROM test_unicode WHERE val = ?", [val])
                assert len(rows) == 1, f"Failed to find: {repr(val)}"
                assert rows[0]["val"] == val, f"Mismatch for: {repr(val)}"

                await conn.execute("DELETE FROM test_unicode WHERE val = ?", [val])

    async def test_binary_blob(self, cluster_address: str) -> None:
        """Binary blobs including null bytes."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_blob")
            await conn.execute("CREATE TABLE test_blob (id INTEGER PRIMARY KEY, data BLOB)")

            blob_values = [
                b"simple",
                b"\x00\x01\x02\x03",
                b"\xff\xfe\xfd",
                bytes(range(256)),
                b"binary\x00with\x00nulls",
            ]

            for val in blob_values:
                await conn.execute("INSERT INTO test_blob (data) VALUES (?)", [val])

                rows = await conn.fetch("SELECT data FROM test_blob ORDER BY id DESC LIMIT 1")
                assert len(rows) == 1
                assert rows[0]["data"] == val, f"Mismatch for blob: {repr(val)}"

    async def test_numeric_types(self, cluster_address: str) -> None:
        """Integer and float edge cases."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_numeric")
            await conn.execute(
                "CREATE TABLE test_numeric "
                "(id INTEGER PRIMARY KEY, int_val INTEGER, float_val REAL)"
            )

            test_cases = [
                (0, 0.0),
                (1, 1.0),
                (-1, -1.0),
                (2147483647, 3.14159265358979),  # Max 32-bit signed
                (-2147483648, -3.14159265358979),  # Min 32-bit signed
                (9223372036854775807, 1e10),  # Max 64-bit signed
                (-9223372036854775808, 1e-10),  # Min 64-bit signed
            ]

            for int_val, float_val in test_cases:
                await conn.execute(
                    "INSERT INTO test_numeric (int_val, float_val) VALUES (?, ?)",
                    [int_val, float_val],
                )

                rows = await conn.fetch(
                    "SELECT int_val, float_val FROM test_numeric ORDER BY id DESC LIMIT 1"
                )
                assert len(rows) == 1
                assert rows[0]["int_val"] == int_val, f"Integer mismatch for {int_val}"
                assert abs(rows[0]["float_val"] - float_val) < 1e-9, (
                    f"Float mismatch for {float_val}"
                )

    async def test_boolean_values(self, cluster_address: str) -> None:
        """Boolean handling (SQLite uses INTEGER 0/1)."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_bool")
            await conn.execute("CREATE TABLE test_bool (id INTEGER PRIMARY KEY, flag INTEGER)")

            await conn.execute("INSERT INTO test_bool (flag) VALUES (?)", [1])
            await conn.execute("INSERT INTO test_bool (flag) VALUES (?)", [0])

            rows = await conn.fetch("SELECT flag FROM test_bool ORDER BY id")
            assert len(rows) == 2
            assert rows[0]["flag"] == 1
            assert rows[1]["flag"] == 0

    async def test_datetime_as_text(self, cluster_address: str) -> None:
        """Datetime stored as ISO8601 text."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_datetime")
            await conn.execute(
                "CREATE TABLE test_datetime (id INTEGER PRIMARY KEY, created_at TEXT)"
            )

            now = datetime.datetime.now()
            iso_string = now.isoformat()

            await conn.execute("INSERT INTO test_datetime (created_at) VALUES (?)", [iso_string])

            rows = await conn.fetch("SELECT created_at FROM test_datetime ORDER BY id DESC LIMIT 1")
            assert len(rows) == 1
            assert rows[0]["created_at"] == iso_string

            parsed = datetime.datetime.fromisoformat(rows[0]["created_at"])
            assert parsed == now

    async def test_null_values(self, cluster_address: str) -> None:
        """NULL handling across column types."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_nulls")
            await conn.execute(
                "CREATE TABLE test_nulls "
                "(id INTEGER PRIMARY KEY, int_col INTEGER, text_col TEXT, "
                "real_col REAL, blob_col BLOB)"
            )

            await conn.execute(
                "INSERT INTO test_nulls (int_col, text_col, real_col, blob_col) "
                "VALUES (?, ?, ?, ?)",
                [None, None, None, None],
            )

            rows = await conn.fetch(
                "SELECT int_col, text_col, real_col, blob_col "
                "FROM test_nulls ORDER BY id DESC LIMIT 1"
            )
            assert len(rows) == 1
            assert rows[0]["int_col"] is None
            assert rows[0]["text_col"] is None
            assert rows[0]["real_col"] is None
            assert rows[0]["blob_col"] is None

    async def test_null_boolean_and_datetime(self, cluster_address: str) -> None:
        """Catches a dqlite bug returning NULL BOOLEAN/DATETIME as False/'' not None."""
        async with await connect(cluster_address) as conn:
            await conn.execute("DROP TABLE IF EXISTS test_null_special")
            await conn.execute(
                "CREATE TABLE test_null_special "
                "(id INTEGER PRIMARY KEY, bool_col BOOLEAN, dt_col DATETIME)"
            )

            await conn.execute(
                "INSERT INTO test_null_special (id, bool_col, dt_col) VALUES (1, NULL, NULL)"
            )
            # Zero values for comparison against the NULLs above.
            await conn.execute(
                "INSERT INTO test_null_special (id, bool_col, dt_col) VALUES (2, 0, '')"
            )

            rows = await conn.fetch(
                "SELECT id, bool_col, dt_col FROM test_null_special ORDER BY id"
            )
            assert len(rows) == 2

            assert rows[0]["bool_col"] is None, (
                f"BOOLEAN NULL should be None, got {rows[0]['bool_col']!r}"
            )
            assert rows[0]["dt_col"] is None, (
                f"DATETIME NULL should be None, got {rows[0]['dt_col']!r}"
            )

            assert rows[1]["bool_col"] is False or rows[1]["bool_col"] == 0
            assert rows[1]["dt_col"] == "" or rows[1]["dt_col"] is None
