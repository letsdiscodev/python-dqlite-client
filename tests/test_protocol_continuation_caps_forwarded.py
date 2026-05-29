"""``DqliteProtocol`` forwards user-supplied continuation caps to the
``MessageDecoder``; otherwise the codec's defaults silently override the
user's intent in both directions (raising the cap, or disabling it)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from dqliteclient.protocol import DqliteProtocol


def _build_protocol(**kwargs: Any) -> DqliteProtocol:
    reader = AsyncMock()
    writer = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return DqliteProtocol(reader, writer, **kwargs)


def test_forwards_explicit_max_total_rows_to_decoder() -> None:
    proto = _build_protocol(max_total_rows=50_000_000)
    assert proto._decoder._max_total_rows == 50_000_000


def test_forwards_explicit_max_continuation_frames_to_decoder() -> None:
    proto = _build_protocol(max_continuation_frames=500_000)
    assert proto._decoder._max_continuation_frames == 500_000


def test_none_max_total_rows_disables_codec_cap() -> None:
    """``None`` means disabled: the codec accepts it and skips the cap check."""
    proto = _build_protocol(max_total_rows=None)
    assert proto._decoder._max_total_rows is None


def test_none_max_continuation_frames_disables_codec_cap() -> None:
    proto = _build_protocol(max_continuation_frames=None)
    assert proto._decoder._max_continuation_frames is None


def test_default_caps_match_protocol_defaults() -> None:
    """With no caps specified, the protocol forwards its defaults, which equal
    the wire package's ``DEFAULT_*`` constants."""
    from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES, DEFAULT_MAX_TOTAL_ROWS

    proto = _build_protocol()
    assert proto._decoder._max_total_rows == DEFAULT_MAX_TOTAL_ROWS
    assert proto._decoder._max_continuation_frames == DEFAULT_MAX_CONTINUATION_FRAMES
