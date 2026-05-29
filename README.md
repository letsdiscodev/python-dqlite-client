# dqlite-client

Async Python client for [dqlite](https://dqlite.io/), Canonical's
distributed SQLite. It connects to a dqlite cluster, finds the current
Raft leader, runs SQL, and pools connections.

The API is inspired by [asyncpg](https://github.com/MagicStack/asyncpg)'s
ergonomics — explicit `connect()` / `create_pool()`, `fetch` / `fetchall`
/ `fetchval`, and context-manager transactions — with a deliberately
simple data model: `fetch` returns `list[dict]`, `fetchall` returns
`list[list]`, `fetchval` returns a scalar. There is no `Record` type.

## Is this the package you want?

Use `dqlite-client` directly when you want a lightweight async client
with no ORM or DB-API layer, or when you need fine-grained control over
cluster bootstrapping and the wire protocol.

If you want a standard PEP 249 driver, SQLAlchemy, Alembic, or most ORMs,
reach for a higher layer instead — see
[The dqlite Python stack](#the-dqlite-python-stack) below.

## Installation

```bash
pip install dqlite-client
```

Requires Python 3.13+.

## Usage

```python
import asyncio
from dqliteclient import connect

async def main():
    conn = await connect("localhost:9001")
    async with conn.transaction():
        await conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.execute("INSERT INTO t (name) VALUES (?)", ["hello"])
        rows = await conn.fetch("SELECT * FROM t")   # list[dict]
        for row in rows:
            print(row)
    await conn.close()

asyncio.run(main())
```

### Connection pool

```python
from dqliteclient import create_pool

pool = await create_pool(["localhost:9001", "localhost:9002", "localhost:9003"])
async with pool.acquire() as conn:
    rows = await conn.fetch("SELECT 1")
```

## The dqlite Python stack

This is one of four layered packages. Each builds on the one below:

| Package | Role |
| --- | --- |
| [sqlalchemy-dqlite](https://github.com/letsdiscodev/sqlalchemy-dqlite) | SQLAlchemy 2.0 dialect |
| [dqlite-dbapi](https://github.com/letsdiscodev/python-dqlite-dbapi) | PEP 249 (DB-API 2.0) driver — sync & async |
| **dqlite-client** — this package | Async wire client — pooling, leader discovery |
| [dqlite-wire](https://github.com/letsdiscodev/python-dqlite-wire) | Wire-protocol codec |

For SQLAlchemy/ORM use, prefer
[sqlalchemy-dqlite](https://github.com/letsdiscodev/sqlalchemy-dqlite) or
[dqlite-dbapi](https://github.com/letsdiscodev/python-dqlite-dbapi).

## Documentation

- [Connection pooling](docs/connection-pooling.md) — sizing, startup, and leader behavior.
- [Deployment: forking & multiprocessing](docs/deployment.md) — read this
  if you use gunicorn, Celery, or `multiprocessing`.
- [Data types & NULL handling](docs/data-types.md) — the row model and a
  server-version gotcha for NULLs in typed columns.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and contribution guidelines.

## License

MIT
