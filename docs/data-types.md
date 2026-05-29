# Data types & NULL handling

## The row model

The fetch methods return plain Python containers — there is no `Record`
type:

- `fetch(...)` → `list[dict]` (one dict per row, keyed by column name)
- `fetchall(...)` → `list[list]` (one list per row)
- `fetchval(...)` → a single scalar value

If you need a richer row object, wrap the results yourself.

## NULL in BOOLEAN/DATETIME columns depends on the server version

A 2026 dqlite server change (upstream commit `f30fc99`, *"query: preserve
SQLITE_NULL type for NULL values"*) changed how a NULL cell is encoded in
columns declared `BOOLEAN`, `DATE`, `DATETIME`, or `TIMESTAMP`:

- **Before** that change: a NULL was sent with the column's coerced type —
  `BOOLEAN` with value `0` (decodes to `False`), or an empty ISO-8601 string
  (decodes to `""`). Indistinguishable on the wire from a real `False` /
  empty string.
- **After** that change: a NULL is sent as SQLite NULL and decodes to
  `None`.

This client decodes faithfully whatever the server sends — the *server*
version drives the behavior, and there is no handshake field that
distinguishes the two. So code like `if row["flag"] is None:` will silently
miss rows against an old cluster, then start matching after the cluster is
upgraded. Check your dqlite cluster version before relying on `is None` for
`BOOLEAN` / `DATETIME` columns.
