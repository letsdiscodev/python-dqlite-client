# Architecture

This page records the design decisions behind `dqlite-client`. User-facing
behaviour is documented in the README and the other pages under `docs/`.

## Layering

```
ConnectionPool      idle queue + size cap, ROLLBACK on return
    │
ClusterClient       leader discovery, admin RPCs, connect-with-retry
    │
DqliteConnection    one wire session: execute / query, transaction tracking
    │
DqliteProtocol      request/response framing, deadlines, continuation caps
    │
dqlitewire          codec
```

| Module | Responsibility |
| --- | --- |
| `protocol.py` | `DqliteProtocol`: handshake, one RPC at a time, read deadlines, row and frame caps |
| `connection.py` | `DqliteConnection`: lifecycle, the `execute` / `fetch` family, `transaction()` |
| `cluster.py` | `ClusterClient`: `find_leader`, `connect`, admin RPCs, redirect policies |
| `pool.py` | `ConnectionPool` |
| `node_store.py` | `NodeInfo`, `NodeStore`, `MemoryNodeStore`, `YamlNodeStore` |
| `sql.py` | lexical SQL helpers shared with the dbapi layer |
| `_validate.py` | `parse_address`, `validate_timeout`, defaults |
| `_dial.py` | TCP dialing with keepalive; the `DialFunc` hook |
| `retry.py` | `retry_with_backoff` |

## Connection lifecycle

- Constructing a `DqliteConnection` validates its arguments and does no I/O.
  `connect()` dials, handshakes and opens the database under `attempt_timeout`
  (with `dial_timeout` nested for the TCP phase).
- **A connection that loses its session is dead.** A transport error, a wire
  decode failure, a leader-change result code, or a cancellation or interrupt
  delivered while a round-trip is in flight all invalidate the connection: the
  transport is closed, `is_connected` becomes `False`, and every further
  operation raises `DqliteConnectionError("Not connected")`. There is no
  reconnect on the same object; open a new one (the pool and the dbapi layer
  do this for you). Server-side SQL errors leave the connection usable.
- `close()` is idempotent. It closes the transport and waits for it, bounded
  by `close_timeout`. `terminate()` is the synchronous variant for paths that
  cannot await; it never raises.
- One operation at a time: a second coroutine calling while another is
  mid-round-trip gets `InterfaceError`. Use one connection per task, or a
  pool.
- A connection binds to the event loop it first runs on and rejects use from
  another loop or after `os.fork()` with `InterfaceError`.
- A connection garbage-collected while still open emits a `ResourceWarning`.

## Transaction tracking

`in_transaction` is a local flag maintained from the statements the
connection sends, and it is deliberately conservative:

| Statement | Effect on the flag |
| --- | --- |
| `BEGIN`, `SAVEPOINT` | set |
| `COMMIT`, `END`, `ROLLBACK` (without `TO`) | clear |
| `RELEASE`, `ROLLBACK TO` | unchanged |
| a server reply whose result code means SQLite rolled back on its own | clear |

The one case where the flag over-reports is releasing the outermost savepoint
opened outside a `BEGIN`: the engine is back in autocommit but the flag stays
set until the next `COMMIT` or `ROLLBACK`. The only cost is one `ROLLBACK`
the server answers with "no transaction is active", which the pool and
`transaction()` treat as success. Multi-statement input is split and every
piece is applied in order.

## Leader discovery

Discovery and redirect verification mirror go-dqlite's connector. One difference:
`ClusterClient.connect` retries a bounded number of attempts (3 by default, jittered
backoff capped at 1 s) instead of retrying until the caller's deadline as go-dqlite does;
pass `max_attempts` / `max_elapsed_seconds` to widen it.

`find_leader` is single-flight per `(trust_server_heartbeat, policy)`: concurrent
callers await the same sweep. A sweep probes the cached leader first, then
every node from the store, shuffled and ordered voters first, with at most
`concurrent_leader_conns` probes in flight. The first probe that yields a
leader wins and the rest are cancelled. A redirect to another node is verified
by asking that node before it is trusted, and every address a peer hands back
is checked against the redirect policy; a rejection raises
`ClusterPolicyError` and is never retried. Probes use a version-only handshake
so they do not consume a server client slot.

`connect()` runs discovery plus `DqliteConnection.connect()` under
`retry_with_backoff`, retrying transport and cluster failures only.

## Pool

`acquire()` hands out an idle connection, opens a new one if the pool is
below `max_size`, or waits, all under one `timeout`. `min_size` is a warm-up
count opened by `initialize()`, not a floor the pool maintains. On return the
connection is rolled back if its flag says a transaction may be open, and
dropped if its transport died. `close()` closes idle connections; connections
still checked out close when they are returned.

## Errors

| Situation | Exception |
| --- | --- |
| dial, handshake, read or write failure; server closed; deadline exceeded | `DqliteConnectionError` (with `code` when a leader-change result code caused it) |
| server `FAILURE` reply to an RPC | `OperationalError(message, code)`; `AmbiguousCommitError` when leadership is lost during `COMMIT` in `transaction()` |
| malformed or unexpected frame | `ProtocolError` (a `DqliteError` and a `dqlitewire.ProtocolError`) |
| no leader found | `ClusterError`, whose message lists each node's failure |
| redirect rejected by policy | `ClusterPolicyError` |
| bind value not encodable | `DataError`; the connection stays usable |
| misuse: closed, wrong loop, after fork, concurrent use | `InterfaceError` |

Wordings the dbapi and SQLAlchemy layers match on and that must stay stable:
`Failed to connect to`, `timed out`, `Not connected`, `Connection closed by
server`, `Could not find leader`, the `wire decode failed` prefix from
`dqlitewire`, and `used after fork`.
