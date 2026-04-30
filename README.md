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

The pool issues `min_size` connection handshakes in parallel during
`initialize()`. All initial connects target whichever node the
cluster client identifies as the current leader, so the leader
serialises its incoming-connection acceptance — `min_size=N` does
NOT speed up startup linearly with N. A balanced default
(`min_size=1` or low single digits) keeps cold-start latency
predictable; raise it only when steady-state concurrency demands
warm connections at engine startup.

## Forking and multiprocessing

Connections, pools, and the cluster client are not safe to use
across `os.fork()`. The library detects fork-after-init and raises
`InterfaceError` from any operation in the child process; the
inherited TCP socket would otherwise be shared with the parent
(writes would interleave on the wire) and asyncio primitives are
bound to the parent's event loop.

Common deployment patterns that fork after import:

- **gunicorn** with `--preload`: workers inherit pools created in
  the parent. Move pool creation into a per-worker `post_fork`
  hook (gunicorn `post_fork` config) instead of the module top
  level.
- **multiprocessing**: child processes must reconstruct
  connections / pools from configuration (addresses, database
  name) rather than receive a parent-built object.
- **Celery prefork pool**: each worker process must create its
  own pool inside the worker init signal, not at module load.

The fork detection is best-effort (pid mismatch); silent
double-FIN on the parent's socket is a real risk if the guard is
bypassed (e.g. via `__new__` to skip `__init__`). Just create the
pool / connection in the child process you intend to use it from.

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
