"""Download–Screen–Delete (single-recording peak disk footprint).

For every recording a backend fetches the small sidecars first (so the channel names can be checked before
any large binary is downloaded), then the main file and *all* of its companion files:

* BrainVision  ``.vhdr`` + ``.vmrk`` + ``.eeg``/``.dat``  (the names in the header's ``DataFile=`` and
  ``MarkerFile=`` lines are honoured)
* EEGLAB       ``.set`` + ``.fdt`` (when the dataset has one)
* single-file  ``.edf``, ``.bdf``, ``.fif``

After the metrics are persisted, :meth:`Backend.cleanup` deletes everything that was fetched for that
recording (called from a ``finally`` block, so also after an error). At any moment only one recording
of the dataset is on disk.

Backends: :class:`HttpsBackend` (pure Python, OpenNeuro's public S3 bucket), :class:`DataladBackend`
(``datalad get`` / ``datalad drop``, exact snapshot pinning) and :class:`LocalBackend` (a BIDS folder that
already exists; never deletes anything).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote

from .discovery import DatasetInfo, RecordingSpec
from .net import Endpoints, Http, HttpError


class DownloadError(Exception):
    """A required file could not be fetched completely."""


class BackendError(Exception):
    """The requested backend is not usable (e.g. DataLad not installed)."""


@dataclass
class FetchSession:
    """Everything fetched for ONE recording, recorded the moment each file arrives, so that cleanup also
    removes partial work after a failure."""
    work_root: Path
    files: list[Path] = field(default_factory=list)
    main: Path | None = None
    sidecars: dict[str, Path] = field(default_factory=dict)
    bytes_downloaded: int = 0
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def local(self, rel: str) -> Path:
        return self.work_root / PurePosixPath(rel)

    @property
    def companions(self) -> list[Path]:
        return [p for p in self.files if p != self.main and p not in self.sidecars.values()]


class Backend:
    name = "base"
    deletes_data = False

    def prepare(self) -> dict[str, Any]:
        return {}

    def new_session(self) -> FetchSession:
        raise NotImplementedError

    def fetch_sidecars(self, spec: RecordingSpec, session: FetchSession) -> None:
        raise NotImplementedError

    def fetch_data(self, spec: RecordingSpec, session: FetchSession) -> None:
        raise NotImplementedError

    def cleanup(self, session: FetchSession) -> dict[str, Any]:
        return {"deleted": 0, "bytes": 0, "errors": [], "kept": True}


# ------------------------------------------------------------------------------------------------ helpers
_VHDR_REF = re.compile(r"^\s*(DataFile|MarkerFile)\s*=\s*(.+?)\s*$", re.I | re.M)


def parse_vhdr_references(vhdr: Path) -> dict[str, str]:
    """``DataFile=`` and ``MarkerFile=`` of a BrainVision header (base names only; paths are not followed)."""
    try:
        text = Path(vhdr).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return {}
    out = {}
    for key, val in _VHDR_REF.findall(text):
        out[key.lower()] = PurePosixPath(val.replace("\\", "/")).name
    return out


def prune_empty_dirs(start: Path, stop_at: Path) -> None:
    """Remove empty directories from ``start`` upwards, never touching ``stop_at`` or anything above it."""
    stop_at = stop_at.resolve()
    d = start.resolve()
    while d != stop_at and stop_at in d.parents:
        try:
            d.rmdir()
        except OSError:
            break
        d = d.parent


# ------------------------------------------------------------------------------------------------ local folder
class LocalBackend(Backend):
    """A BIDS folder that is already on disk. Nothing is copied and nothing is ever deleted."""
    name = "local"
    deletes_data = False

    def __init__(self, bids_dir: Path) -> None:
        self.root = Path(bids_dir)

    def new_session(self) -> FetchSession:
        return FetchSession(work_root=self.root)

    def fetch_sidecars(self, spec: RecordingSpec, session: FetchSession) -> None:
        for key, rel in spec.sidecars.items():
            p = session.local(rel)
            if p.exists():
                session.sidecars[key] = p

    def fetch_data(self, spec: RecordingSpec, session: FetchSession) -> None:
        main = session.local(spec.path)
        if not main.exists():
            raise DownloadError(f"file not found: {main}")
        session.main = main
        session.files.append(main)
        for rel in spec.companions:
            p = session.local(rel)
            if p.exists():
                session.files.append(p)
            else:
                session.warnings.append(f"companion file missing: {rel}")
        if spec.fmt == "vhdr":
            refs = parse_vhdr_references(main)
            for key in ("datafile", "markerfile"):
                name = refs.get(key)
                if name and not (main.parent / name).exists():
                    raise DownloadError(f"BrainVision {key} '{name}' referenced by {main.name} is missing")


# ------------------------------------------------------------------------------------------------ HTTPS / S3
class HttpsBackend(Backend):
    """Pure-Python downloads from OpenNeuro's public S3 bucket. No DataLad / git-annex needed.

    The bucket serves the newest snapshot of a dataset (older snapshots are not addressable this way); use the
    DataLad backend for an exactly pinned version.
    """
    name = "https"
    deletes_data = True

    def __init__(self, dataset_id: str, work_dir: Path, http: Http | None = None, endpoints: Endpoints | None = None,
                 snapshot: str | None = None) -> None:
        self.ds = dataset_id
        self.work = Path(work_dir)
        self.http = http or Http()
        self.ep = endpoints or Endpoints.from_env()
        self.snapshot = snapshot

    def prepare(self) -> dict[str, Any]:
        self.work.mkdir(parents=True, exist_ok=True)
        info: dict[str, Any] = {"method": "https", "source": self.ep.s3_base}
        if self.snapshot:
            info["warning"] = (f"HTTPS downloads the newest OpenNeuro snapshot, which is not guaranteed to be "
                               f"{self.snapshot}. Use --download-method datalad for an exactly pinned version.")
        return info

    def new_session(self) -> FetchSession:
        return FetchSession(work_root=self.work)

    def _url(self, rel: str) -> str:
        return f"{self.ep.s3_base}/{self.ds}/{quote(rel, safe='/')}"

    def _get(self, rel: str, session: FetchSession) -> Path:
        dst = session.local(rel)
        session.files.append(dst)          # registered BEFORE the download: a partial file is cleaned up as well
        session.bytes_downloaded += self.http.download(self._url(rel), dst)
        return dst

    def fetch_sidecars(self, spec: RecordingSpec, session: FetchSession) -> None:
        for key, rel in spec.sidecars.items():
            try:
                session.sidecars[key] = self._get(rel, session)
            except HttpError as e:
                session.warnings.append(f"sidecar {rel} not fetched ({e})")

    def fetch_data(self, spec: RecordingSpec, session: FetchSession) -> None:
        try:
            session.main = self._get(spec.path, session)
        except HttpError as e:
            raise DownloadError(f"main file {spec.path}: {e}") from e
        d = PurePosixPath(spec.path).parent
        needed: dict[str, bool] = {c: True for c in spec.companions}     # companion path -> required
        if spec.fmt == "vhdr":
            refs = parse_vhdr_references(session.main)
            stem = PurePosixPath(spec.path).name.split(".", 1)[0]
            for key in ("datafile", "markerfile"):
                name = refs.get(key)
                if name:
                    needed.setdefault(str(d / name), True)
            if not refs:
                session.warnings.append("could not read DataFile/MarkerFile from the .vhdr; using BIDS-named companions")
                for ext in (".vmrk", ".eeg"):
                    needed.setdefault(str(d / f"{stem}{ext}"), ext == ".eeg")
        for rel, required in needed.items():
            try:
                self._get(rel, session)
            except HttpError as e:
                if required:
                    raise DownloadError(f"companion file {rel}: {e}") from e
                session.warnings.append(f"optional companion {rel} not fetched ({e})")
        if session.main.stat().st_size == 0:
            raise DownloadError(f"{spec.path} downloaded as an empty file")

    def cleanup(self, session: FetchSession) -> dict[str, Any]:
        deleted, size, errors = 0, 0, []
        dirs = set()
        for p in sorted(set(session.files), key=str, reverse=True):
            for candidate in (p, p.with_name(p.name + ".part")):
                try:
                    if candidate.exists() or candidate.is_symlink():
                        size += candidate.stat().st_size if candidate.exists() else 0
                        candidate.unlink()
                        deleted += 1
                except OSError as e:
                    errors.append(f"{candidate}: {e}")
            dirs.add(p.parent)
        for d in sorted(dirs, key=lambda x: len(x.parts), reverse=True):
            prune_empty_dirs(d, session.work_root)
        session.files.clear()
        return {"deleted": deleted, "bytes": size, "errors": errors, "kept": False}


# ------------------------------------------------------------------------------------------------ DataLad
class DataladBackend(Backend):
    """``datalad get`` / ``datalad drop`` inside a clone of ``OpenNeuroDatasets/<id>`` pinned to a snapshot tag."""
    name = "datalad"
    deletes_data = True

    def __init__(self, dataset_id: str, repo_dir: Path, tag: str | None = None,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.ds, self.repo, self.tag, self.run = dataset_id, Path(repo_dir), tag, runner

    @staticmethod
    def available() -> bool:
        return shutil.which("datalad") is not None and shutil.which("git-annex") is not None

    def _run(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        proc = self.run(cmd, capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise DownloadError(f"{' '.join(cmd[:3])} failed: {(proc.stderr or proc.stdout or '').strip()[:300]}")
        return proc

    def prepare(self) -> dict[str, Any]:
        if not (self.repo / ".git").exists():
            self.repo.parent.mkdir(parents=True, exist_ok=True)
            self._run(["datalad", "install", "-s", f"https://github.com/OpenNeuroDatasets/{self.ds}", str(self.repo)])
        if self.tag:
            self._run(["git", "-C", str(self.repo), "checkout", "-q", self.tag])
        commit = self._run(["git", "-C", str(self.repo), "rev-parse", "HEAD"]).stdout.strip()
        return {"method": "datalad", "tag": self.tag, "commit": commit[:12]}

    def new_session(self) -> FetchSession:
        return FetchSession(work_root=self.repo)

    def _get_paths(self, rels: list[str], session: FetchSession) -> None:
        paths = [str(session.local(r)) for r in rels]
        session.files += [session.local(r) for r in rels]
        self._run(["datalad", "get", "-d", str(self.repo), *paths])

    def fetch_sidecars(self, spec: RecordingSpec, session: FetchSession) -> None:
        for key, rel in spec.sidecars.items():
            try:
                self._get_paths([rel], session)
                session.sidecars[key] = session.local(rel)
            except DownloadError as e:
                session.warnings.append(f"sidecar {rel} not fetched ({e})")

    def fetch_data(self, spec: RecordingSpec, session: FetchSession) -> None:
        self._get_paths([spec.path, *spec.companions], session)
        session.main = session.local(spec.path)
        if spec.fmt == "vhdr":
            d = PurePosixPath(spec.path).parent
            refs = parse_vhdr_references(session.main)
            extra = [str(d / n) for n in refs.values() if str(d / n) not in spec.companions]
            if extra:
                self._get_paths(extra, session)

    def cleanup(self, session: FetchSession) -> dict[str, Any]:
        files = [p for p in dict.fromkeys(session.files)]
        errors = []
        if files:
            proc = self.run(["datalad", "drop", "-d", str(self.repo), "--what", "filecontent", *map(str, files)],
                            capture_output=True, text=True)
            if proc.returncode != 0:
                errors.append((proc.stderr or proc.stdout or "datalad drop failed").strip()[:300])
        n = len(files)
        session.files.clear()
        return {"deleted": n if not errors else 0, "bytes": 0, "errors": errors, "kept": False}


# ------------------------------------------------------------------------------------------------ factory
def make_backend(method: str, info: DatasetInfo, work_dir: Path, repo_dir: Path, snapshot: str | None = None,
                 http: Http | None = None, endpoints: Endpoints | None = None) -> Backend:
    """``auto`` prefers DataLad when it and git-annex are installed, otherwise pure-Python HTTPS."""
    if info.source == "local":
        return LocalBackend(info.bids_dir)               # type: ignore[arg-type]
    method = method.lower()
    if method not in ("auto", "https", "datalad"):
        raise BackendError(f"unknown download method: {method}")
    if method == "datalad" and not DataladBackend.available():
        raise BackendError("--download-method datalad needs 'datalad' and 'git-annex' on the PATH "
                           "(or use --download-method https)")
    if method == "datalad" or (method == "auto" and DataladBackend.available()):
        return DataladBackend(info.dataset_id, repo_dir, tag=snapshot or info.snapshot)
    return HttpsBackend(info.dataset_id, work_dir, http=http, endpoints=endpoints, snapshot=snapshot or info.snapshot)
