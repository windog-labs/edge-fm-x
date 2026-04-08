from __future__ import annotations

import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TMP_ROOT = PROJECT_ROOT / ".tmp_codex" / "tmp"


def ensure_repo_tmp_root() -> Path:
    tmp_root = Path(os.environ.get("TMPDIR", DEFAULT_TMP_ROOT))
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root


def make_temp_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(ensure_repo_tmp_root())))
