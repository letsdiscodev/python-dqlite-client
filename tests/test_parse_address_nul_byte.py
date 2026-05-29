"""``parse_address`` rejects NUL-containing inputs with a specific diagnostic
(the NUL offset) rather than the generic shape-failure message."""

import pytest

from dqliteclient.connection import parse_address


@pytest.mark.parametrize(
    ("address", "expected_offset"),
    [
        ("host\x00:9001", 4),
        ("\x00host:9001", 0),
        ("host:\x009001", 5),
        ("host:9001\x00", 9),
        ("\x00", 0),
    ],
)
def test_parse_address_rejects_nul_byte_with_specific_diagnostic(
    address: str, expected_offset: int
) -> None:
    with pytest.raises(ValueError, match=r"contains NUL byte at offset"):
        parse_address(address)


def test_parse_address_nul_diagnostic_reports_correct_offset() -> None:
    try:
        parse_address("abc\x00def:9001")
    except ValueError as e:
        assert "offset 3" in str(e)
    else:
        raise AssertionError("expected ValueError on NUL-containing address")


def test_parse_address_clean_input_still_parses() -> None:
    host, port = parse_address("host:9001")
    assert host == "host"
    assert port == 9001
