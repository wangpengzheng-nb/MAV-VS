from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable


class SecurityError(ValueError):
    pass


def ensure_within(path: str | Path, roots: Iterable[str | Path], *, must_exist: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = [Path(root).expanduser().resolve() for root in roots]
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise SecurityError(f"path is outside allowed roots: {resolved}")
    if must_exist and not resolved.exists():
        raise SecurityError(f"path does not exist: {resolved}")
    return resolved


def run_argv(argv: list[str], *, cwd: str | Path, timeout: int, log_path: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    if not argv or any("\x00" in item for item in argv):
        raise SecurityError("invalid command arguments")
    result = subprocess.run(
        argv,
        cwd=Path(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        shell=False,
    )
    if log_path:
        Path(log_path).write_text(
            f"COMMAND: {argv!r}\nEXIT: {result.returncode}\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}",
            encoding="utf-8",
        )
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

