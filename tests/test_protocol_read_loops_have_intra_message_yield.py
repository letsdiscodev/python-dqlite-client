"""Pin: ``DqliteProtocol._read_response`` and
``_read_continuation`` each include an ``await asyncio.sleep(0)``
inside their read-and-feed loop so a fast-burst server cannot pin
the event loop for the whole multi-chunk message decode.

The two sibling drain loops (``_drain_continuations`` per-frame
yield, ``_interrupt`` drain yield) already follow this discipline.
The chunk-feed loop is the parity gap one layer down.

The hazard is bounded today by asyncio's private
``_DEFAULT_LIMIT = 64 KiB``, which caps iteration count at ~16
between yields. That bound is not a contract — a future
StreamReader limit bump or a large dump / continuation response
makes the loop monopolise the loop thread for the full chunk-
stream. The cooperative yield removes the dependency on
asyncio's private constants.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dqliteclient import protocol as protocol_mod


def _read_response_source() -> str:
    return textwrap.dedent(inspect.getsource(protocol_mod.DqliteProtocol._read_response))


def _read_continuation_source() -> str:
    return textwrap.dedent(inspect.getsource(protocol_mod.DqliteProtocol._read_continuation))


def _has_asyncio_sleep_zero_in_while(src: str) -> bool:
    """Walk the AST and look for ``await asyncio.sleep(0)`` inside
    any ``while`` loop body.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Await):
                continue
            call = inner.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "sleep"
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
                and len(call.args) == 1
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == 0
            ):
                return True
    return False


def test_read_response_loop_yields_cooperatively() -> None:
    src = _read_response_source()
    assert _has_asyncio_sleep_zero_in_while(src), (
        "DqliteProtocol._read_response's read-and-feed loop must "
        "include an explicit ``await asyncio.sleep(0)`` so a "
        "fast-burst server prefetching the message body does not "
        "starve sibling coroutines. The hazard is bounded today by "
        "asyncio's private StreamReader limit; the explicit yield "
        "removes the dependency on that private detail."
    )


def test_read_continuation_loop_yields_cooperatively() -> None:
    src = _read_continuation_source()
    assert _has_asyncio_sleep_zero_in_while(src), (
        "DqliteProtocol._read_continuation's read-and-feed loop must "
        "include an explicit ``await asyncio.sleep(0)`` for the same "
        "reason as ``_read_response``."
    )
