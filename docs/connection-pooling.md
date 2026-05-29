# Connection pooling

```python
from dqliteclient import create_pool

pool = await create_pool(["localhost:9001", "localhost:9002", "localhost:9003"])
async with pool.acquire() as conn:
    rows = await conn.fetch("SELECT 1")
```

## Sizing and startup

The pool opens `min_size` connections in parallel during `initialize()`.
All initial connects target whichever node the cluster client identifies
as the current Raft leader, so the **leader serializes** its
incoming-connection acceptance — raising `min_size` does **not** speed up
startup linearly with N.

A low default (`min_size=1`, or a small single digit) keeps cold-start
latency predictable. Raise it only when steady-state concurrency genuinely
needs warm connections at engine startup.

## Where pooling lives in the stack

This pool is for direct `dqlite-client` use. If you use
[dqlite-dbapi](https://github.com/letsdiscodev/python-dqlite-dbapi) or
[sqlalchemy-dqlite](https://github.com/letsdiscodev/sqlalchemy-dqlite),
pooling is handled one layer up (SQLAlchemy's `QueuePool` and its async
siblings), and this client-level pool is not used.
