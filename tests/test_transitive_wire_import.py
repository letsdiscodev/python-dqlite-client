"""``import dqliteclient`` must transitively ``import dqlitewire`` at
module-load time: the wire layer's free-threading guard is the only one,
and downstream packages rely on inheriting it via top-level import.

Runs in a subprocess so the sys.modules manipulation does not pollute the
in-process module cache for other tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_dqlitewire_loaded_after_dqliteclient_import() -> None:
    repo_src = Path(__file__).resolve().parent.parent / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    snippet = """
        import sys
        assert "dqlitewire" not in sys.modules, (
            "test setup error: dqlitewire was pre-loaded"
        )
        import dqliteclient  # noqa: F401
        if "dqlitewire" not in sys.modules:
            print("FAIL: dqliteclient did not transitively load dqlitewire", flush=True)
            sys.exit(1)
        print("OK", flush=True)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"transitive-wire-import pin failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
