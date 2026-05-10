"""Pin: ``import dqliteclient`` triggers ``import dqlitewire`` at
module-load time.

The wire-layer free-threading guard lives only in
``python-dqlite-wire/src/dqlitewire/__init__.py``; the downstream
packages (client, dbapi, sqlalchemy-dqlite) deliberately do not
re-implement the guard and rely on transitive top-level imports.

If a future refactor lazifies the wire import (e.g. to break a
circular), the guard inheritance breaks at this entry point: a
free-threaded interpreter could load ``dqliteclient`` without
firing the wire-layer ImportError. CI pin against that drift.

The test runs in a subprocess so manipulating ``sys.modules`` does
not pollute the in-process module cache for other tests.
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
        # Sanity: dqlitewire NOT pre-loaded.
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
