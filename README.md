# dqlite-client

Async Python client for [dqlite](https://dqlite.io/), following asyncpg patterns.

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

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and contribution guidelines.

## License

MIT
