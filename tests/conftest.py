import os
import sys
import tempfile

import pytest

# Ensure the repo root is importable when pytest runs from any CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(autouse=True)
def _sandbox_history_path(monkeypatch, tmp_path_factory):
    """Redirect the global history.jsonl into a temp dir so tests never write to ~."""
    sandbox = tmp_path_factory.mktemp("mopa_history")
    monkeypatch.setenv("MOPA_HEIGHTMAP_HISTORY", str(sandbox / "history.jsonl"))
