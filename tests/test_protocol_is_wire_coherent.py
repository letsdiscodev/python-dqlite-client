"""Pin: ``DqliteProtocol.is_wire_coherent`` routes through the
underlying decoder's ``is_poisoned`` flag.

Existing pool-reset tests stub ``protocol.is_wire_coherent`` as a
boolean attribute on a mock; the real one-line forwarder
(``return not self._decoder.is_poisoned``) at
``protocol.py:190`` was never exercised against a real ``Decoder``
instance. Without a direct test, a future refactor that swapped the
underlying field name (e.g. ``is_poisoned`` → ``is_corrupt``) would
silently break every pool-reset short-circuit caller.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.exceptions import DecodeError


def _make_protocol() -> DqliteProtocol:
    """Construct a ``DqliteProtocol`` against mock reader/writer.

    The protocol's ``is_wire_coherent`` property reads from
    ``self._decoder``, which is a real ``MessageDecoder`` instance —
    so the test exercises the real getter.
    """
    reader = MagicMock(spec=asyncio.StreamReader)
    writer = MagicMock(spec=asyncio.StreamWriter)
    return DqliteProtocol(reader=reader, writer=writer)


def test_is_wire_coherent_true_when_decoder_not_poisoned() -> None:
    proto = _make_protocol()
    # Fresh decoder: not poisoned. The forwarder returns True.
    assert proto._decoder.is_poisoned is False
    assert proto.is_wire_coherent is True


def test_is_wire_coherent_false_when_decoder_poisoned() -> None:
    proto = _make_protocol()
    # Poison the underlying decoder. The forwarder returns False.
    proto._decoder._buffer.poison(DecodeError("forced poison for test"))
    assert proto._decoder.is_poisoned is True
    assert proto.is_wire_coherent is False
