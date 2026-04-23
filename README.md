# dqlite-client

Async Python client for [dqlite](https://dqlite.io/). The API shape
(explicit `connect()` / `create_pool()`, `fetch` / `fetchall` /
`fetchval`, context-manager-driven transactions) is inspired by
asyncpg's ergonomics, but the data model is simpler: `fetch` returns
`list[dict]`, `fetchall` returns `list[list]`, `fetchval` returns a
scalar. There is no Record type — callers who need the asyncpg
Record surface should wrap the rows explicitly.

## Installation

```bash
pip install dqlite-client
```

## Usage

```python
import asyncio
from dqliteclient import connect

async def main():
    conn = await connect("localhost:9001")
    async with conn.transaction():
        await conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO test (name) VALUES (?)", ["hello"])
        rows = await conn.fetch("SELECT * FROM test")
        for row in rows:
            print(row)
    await conn.close()

asyncio.run(main())
```

## Connection Pooling

```python
from dqliteclient import create_pool

pool = await create_pool(["localhost:9001", "localhost:9002", "localhost:9003"])
async with pool.acquire() as conn:
    rows = await conn.fetch("SELECT 1")
```

## Layering

`dqlite-client` is the low-level async wire client. Most applications
should reach for one of the higher-level packages instead:

- **`dqlite-dbapi`** — PEP 249–compliant wrapper (sync *or* async).
  Plug-and-play with SQLAlchemy, Alembic, most ORMs.
- **`sqlalchemy-dqlite`** — SQLAlchemy 2.0 dialect, built on top of
  `dqlite-dbapi`.

Use `dqlite-client` directly when you need fine-grained control over
the wire protocol: custom cluster bootstrapping, explicit message
decoding, or building a new high-level driver.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and contribution guidelines.

## License

MIT
