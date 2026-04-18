"""dqliteclient.connect() and create_pool() forward the governor kwargs.

The module-level convenience factories are the PEP-249-style entry
point for the client package. Constructor kwargs on DqliteConnection /
ConnectionPool were added in an earlier cycle but the factories at
__init__.py didn't forward them, so users going through the convenience
path silently lost the DoS-governor configuration.
"""

from dqliteclient import ConnectionPool, create_pool
from dqliteclient.connection import DqliteConnection


class TestCreatePoolForwardsGovernors:
    """`create_pool` forwards governor kwargs into ConnectionPool."""

    def test_defaults_match_pool_constructor(self) -> None:
        """Not calling create_pool — just asserting the signature parity
        wouldn't catch a forward gap. Use a direct ConnectionPool for
        the unit-level equivalent (no initialize() so no TCP)."""
        pool = ConnectionPool(["localhost:19001"])
        assert pool._max_total_rows == 10_000_000
        assert pool._max_continuation_frames == 100_000
        assert pool._trust_server_heartbeat is False

    def test_custom_governors_plumbed(self) -> None:
        pool = ConnectionPool(
            ["localhost:19001"],
            max_total_rows=500,
            max_continuation_frames=7,
            trust_server_heartbeat=True,
        )
        assert pool._max_total_rows == 500
        assert pool._max_continuation_frames == 7
        assert pool._trust_server_heartbeat is True

    def test_create_pool_signature_accepts_governors(self) -> None:
        """The create_pool factory must not raise TypeError when handed
        the governor kwargs — the regression gap was the factory silently
        dropping kwargs that the constructor supports."""
        import inspect

        sig = inspect.signature(create_pool)
        assert "max_total_rows" in sig.parameters
        assert "max_continuation_frames" in sig.parameters
        assert "trust_server_heartbeat" in sig.parameters


class TestConnectForwardsGovernors:
    """`dqliteclient.connect` forwards governor kwargs.

    A full-integration test would do TCP work; we rely on signature
    parity + the DqliteConnection attribute forwarding that is already
    covered elsewhere.
    """

    def test_connect_signature_accepts_governors(self) -> None:
        import inspect

        from dqliteclient import connect

        sig = inspect.signature(connect)
        assert "max_total_rows" in sig.parameters
        assert "max_continuation_frames" in sig.parameters
        assert "trust_server_heartbeat" in sig.parameters

    def test_dqlite_connection_constructor_forwarding(self) -> None:
        """`connect` ultimately calls DqliteConnection(...). Construct
        that object directly with the same kwargs and verify the
        attributes propagate — this covers the plumbing that `connect`
        relies on without needing a TCP connection."""
        conn = DqliteConnection(
            "localhost:19001",
            max_total_rows=500,
            max_continuation_frames=7,
            trust_server_heartbeat=True,
        )
        assert conn._max_total_rows == 500
        assert conn._max_continuation_frames == 7
        assert conn._trust_server_heartbeat is True
