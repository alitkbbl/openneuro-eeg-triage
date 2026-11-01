"""Small helpers shared by all modules (no scientific logic here)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def now_iso() -> str:
    """Current UTC time, ISO 8601, second precision."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas / Path objects to plain JSON types. NaN and Inf become ``None``."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if hasattr(obj, "item") and callable(obj.item):           # other numpy-like scalars
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def nan_if_none(v: Any) -> float:
    """``None`` (the JSON form of NaN) back to ``float('nan')``."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON so that a crash can never leave a half-written file."""
    atomic_write_text(path, json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def natural_key(text: str) -> list[Any]:
    """Sort key that orders ``sub-2`` before ``sub-10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def human_seconds(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


class Logger:
    """Minimal non-interactive logger: plain lines on stdout, warnings on stderr. Never asks anything."""

    def __init__(self, quiet: bool = False, stream=None, err=None) -> None:
        self.quiet = quiet
        self.stream = stream or sys.stdout
        self.err = err or sys.stderr

    def info(self, msg: str = "") -> None:
        if not self.quiet:
            print(msg, file=self.stream, flush=True)

    def step(self, msg: str) -> None:
        self.info(f"  {msg}")

    def warn(self, msg: str) -> None:
        print(f"  ! {msg}", file=self.err, flush=True)

    def error(self, msg: str) -> None:
        print(f"ERROR: {msg}", file=self.err, flush=True)


def chunked(items: Iterable[Any], n: int) -> Iterable[list[Any]]:
    buf: list[Any] = []
    for it in items:
        buf.append(it)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


Callback = Callable[..., None]
