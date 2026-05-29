"""Pin: ``DqliteProtocol.is_wire_coherent`` forwards the underlying
decoder's ``is_poisoned`` flag, exercised against a real decoder.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from dqliteclient.protocol import DqliteProtocol
from dqlitewire.exceptions import DecodeError


def _make_protocol() -> DqliteProtocol:
    """Build a protocol whose ``_decoder`` is a real ``MessageDecoder``."""
    reader = MagicMock(spec=asyncio.StreamReader)
    writer = MagicMock(spec=asyncio.StreamWriter)
    return DqliteProtocol(reader=reader, writer=writer)


def test_is_wire_coherent_true_when_decoder_not_poisoned() -> None:
    proto = _make_protocol()
    assert proto._decoder.is_poisoned is False
    assert proto.is_wire_coherent is True


def test_is_wire_coherent_false_when_decoder_poisoned() -> None:
    proto = _make_protocol()
    proto._decoder._buffer.poison(DecodeError("forced poison for test"))
    assert proto._decoder.is_poisoned is True
    assert proto.is_wire_coherent is False
