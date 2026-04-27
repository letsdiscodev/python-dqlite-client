"""Pin: ``DqliteProtocol`` forwards user-supplied continuation caps to
the underlying ``MessageDecoder``.

The wire-layer codec enforces ``max_continuation_frames`` and
``max_total_rows`` per-decoder. If the protocol layer constructs the
codec with no kwargs, the codec's defaults (10M rows, 100k frames)
silently override the user's intent — both directions:

- A user bumping ``max_total_rows`` above 10M would see ``DecodeError``
  from the codec instead of the client-layer cap.
- A user passing ``None`` ("disabled") would still be capped at 10M
  by the codec.

Pin both directions.
"""

from __future__ import annotations

import sys
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
    """``None`` at the protocol layer means "client-layer disabled";
    the codec must not enforce a stricter cap. Forward as
    ``sys.maxsize`` so the codec never trips."""
    proto = _build_protocol(max_total_rows=None)
    assert proto._decoder._max_total_rows == sys.maxsize


def test_none_max_continuation_frames_disables_codec_cap() -> None:
    proto = _build_protocol(max_continuation_frames=None)
    assert proto._decoder._max_continuation_frames == sys.maxsize


def test_default_caps_match_protocol_defaults() -> None:
    """Negative pin: with no caps specified, the protocol forwards its
    own defaults to the codec. The codec's defaults equal the wire-
    package's exported ``DEFAULT_*`` constants, so the values match."""
    from dqlitewire import DEFAULT_MAX_CONTINUATION_FRAMES, DEFAULT_MAX_TOTAL_ROWS

    proto = _build_protocol()
    assert proto._decoder._max_total_rows == DEFAULT_MAX_TOTAL_ROWS
    assert proto._decoder._max_continuation_frames == DEFAULT_MAX_CONTINUATION_FRAMES
