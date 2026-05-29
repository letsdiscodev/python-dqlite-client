# Deployment: forking & multiprocessing

Connections, pools, and the cluster client are **not safe to use across
`os.fork()`.** A forked child inherits the parent's TCP socket (writes
from both would interleave on the wire) and the asyncio primitives are
bound to the parent's event loop.

The library detects fork-after-init and raises `InterfaceError` from any
operation in the child process. Detection is best-effort (it compares the
PID), so the rule to follow is simple:

> **Create the pool/connection in the process that will use it.**

## Common patterns that fork after import

- **gunicorn with `--preload`**: workers inherit objects created in the
  parent. Create your pool in a per-worker `post_fork` hook instead of at
  module top level.
- **`multiprocessing`**: child processes must rebuild connections/pools
  from configuration (addresses, database name), not receive a
  parent-built object.
- **Celery prefork pool**: each worker process must create its own pool in
  the worker-init signal, not at module load.

If you bypass the guard (for example by constructing via `__new__` to skip
`__init__`), a silent double-FIN on the parent's socket is a real risk.
Don't share these objects across a fork.
