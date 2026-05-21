"""Pin: every public admin-RPC method on :class:`ClusterClient`
runs the ``_check_pid`` fork guard (directly or via
:meth:`find_leader`) and therefore must name :class:`InterfaceError`
in its docstring's ``Raises:`` block.

Pre-fix the docstrings enumerated ``ClusterError`` /
``OperationalError`` / ``ProtocolError`` (sometimes also
``DqliteConnectionError``) but omitted :class:`InterfaceError`. A
defensive admin-RPC wrapper writing ``except (ClusterError,
OperationalError, ProtocolError):`` propagated the undocumented
``InterfaceError`` to the caller — surprising at runtime, doubly
surprising in IDE-hover renderings of the docstring.

``_check_pid``'s own docstring already promises ``InterfaceError``;
this pin keeps the propagation visible to callers reading the
public methods' documentation.
"""

from __future__ import annotations

import pytest

from dqliteclient.cluster import ClusterClient


@pytest.mark.parametrize(
    "method_name",
    [
        "find_leader",
        "describe",
        "set_weight",
        "transfer_leadership",
        "cluster_info",
        "add_node",
        "assign_role",
        "remove_node",
        "leader_info",
        "dump",
        "open_admin_connection",
    ],
)
def test_admin_method_docstring_names_interface_error(method_name: str) -> None:
    """Each method routes through ``_check_pid`` (directly or via
    ``find_leader``); ``InterfaceError`` must appear in the docstring
    so callers can write correct except arms for the fork-after-init
    case."""
    method = getattr(ClusterClient, method_name)
    doc = method.__doc__ or ""
    assert "InterfaceError" in doc, (
        f"{method_name} routes through ``_check_pid`` (or "
        f"``find_leader`` which does) — InterfaceError must be in "
        f"its docstring's Raises block."
    )
