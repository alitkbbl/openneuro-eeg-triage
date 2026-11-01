"""Pre-flight discovery: what is in a dataset, *without* downloading any EEG binary.

* OpenNeuro dataset id  -> the GitHub mirror ``OpenNeuroDatasets/<id>`` (git tree of an exact snapshot tag);
  fallback: the public S3 bucket listing. Small metadata files (``dataset_description.json``,
  ``participants.tsv``) are read directly.
* Local BIDS directory  -> a plain directory walk.

Both produce the same :class:`DatasetInfo` (subjects, sessions, tasks, runs, formats, companion files and
sidecars of every recording), so everything downstream is identical.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

from .config import COMPANION_EXTENSIONS, SUPPORTED_EXTENSIONS
from .net import Endpoints, Http, HttpError
from .utils import natural_key

SKIP_TOP_DIRS = {"derivatives", "sourcedata", "code", "stimuli", ".git", ".datalad", "phenotype", "docs"}
OPENNEURO_ID_RE = re.compile(r"^ds\d{6}$", re.I)


class DiscoveryError(Exception):
    """The dataset could not be inspected (network, rate limit, bad snapshot ...)."""


class DatasetNotFoundError(DiscoveryError):
    """No dataset with this id exists on OpenNeuro."""


class NoEEGDataError(DiscoveryError):
    """The dataset exists but contains no BIDS EEG recordings in a supported format."""


class SelectionError(ValueError):
    """The requested subjects / sessions / tasks do not match anything in the dataset."""


# ------------------------------------------------------------------------------------------------ BIDS names
def parse_bids_name(filename: str) -> dict[str, Any] | None:
    """``sub-01_ses-1_task-rest_run-2_eeg.edf`` -> entities, suffix, extension (``None`` if not BIDS-like)."""
    base = PurePosixPath(filename).name
    if "." not in base:
        return None
    stem, ext = base.split(".", 1)
    ext = "." + ext
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    suffix, ents = parts[-1], {}
    for p in parts[:-1]:
        if "-" not in p:
            return None
        k, v = p.split("-", 1)
        ents[k] = v
    return {"entities": ents, "suffix": suffix, "ext": ext.lower(), "stem": stem}


@dataclass
class RecordingSpec:
    """One EEG recording and everything that has to be fetched with it."""
    subject: str                     # "sub-001"
    session: str | None              # "ses-1" or None for single-session layouts
    task: str | None
    acq: str | None
    run: str | None
    fmt: str                         # "edf", "bdf", "vhdr", "set", "fif"
    path: str                        # BIDS-relative POSIX path of the main file
    size: int | None = None
    companions: list[str] = field(default_factory=list)       # .vmrk/.eeg/.dat, .fdt (paths)
    sidecars: dict[str, str] = field(default_factory=dict)    # "json", "channels", "events" -> path
    entities: dict[str, str] = field(default_factory=dict)

    @property
    def rec_id(self) -> str:
        return PurePosixPath(self.path).name.split(".", 1)[0].removesuffix("_eeg")

    @property
    def directory(self) -> str:
        return str(PurePosixPath(self.path).parent)

    def to_dict(self) -> dict[str, Any]:
        return {"rec_id": self.rec_id, "subject": self.subject, "session": self.session, "task": self.task,
                "acq": self.acq, "run": self.run, "format": self.fmt, "path": self.path, "size": self.size,
                "companions": self.companions, "sidecars": self.sidecars}


@dataclass
class DatasetInfo:
    source: str                                  # "openneuro" | "local"
    dataset_id: str                              # "ds005385" or the directory name
    key: str                                     # identity stored in the output directory
    title: str
    description: dict[str, Any]
    snapshot: str | None
    recordings: list[RecordingSpec]
    participants: dict[str, dict[str, str]]
    bids_dir: Path | None = None
    discovery_method: str = ""
    modalities_seen: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def subjects(self) -> list[str]:
        return sorted({r.subject for r in self.recordings}, key=natural_key)

    @property
    def sessions(self) -> list[str]:
        return sorted({r.session for r in self.recordings if r.session}, key=natural_key)

    @property
    def tasks(self) -> list[str]:
        return sorted({r.task for r in self.recordings if r.task}, key=natural_key)

    @property
    def formats(self) -> dict[str, int]:
        return dict(Counter(r.fmt for r in self.recordings))

    @property
    def total_size(self) -> int | None:
        sizes = [r.size for r in self.recordings]
        return sum(sizes) if sizes and all(s is not None for s in sizes) else None

    def summary_lines(self) -> list[str]:
        from .utils import human_bytes
        fmt = ", ".join(f"{k} x{v}" for k, v in sorted(self.formats.items()))
        lines = [
            f"Dataset      : {self.dataset_id}  ({self.title or 'no title'})",
            f"Source       : {self.source}" + (f", snapshot {self.snapshot}" if self.snapshot else "")
            + (f", discovered via {self.discovery_method}" if self.discovery_method else ""),
            f"Subjects     : {len(self.subjects)} with EEG data"
            + (f" ({len(self.participants)} rows in participants.tsv)" if self.participants else ""),
            f"Sessions     : {', '.join(self.sessions) if self.sessions else 'single-session layout'}",
            f"Tasks        : {', '.join(self.tasks) if self.tasks else 'none declared'}",
            f"Formats      : {fmt}",
            f"Recordings   : {len(self.recordings)}"
            + (f" ({human_bytes(self.total_size)} of EEG data)" if self.total_size else ""),
        ]
        for w in self.warnings:
            lines.append(f"Note         : {w}")
        return lines


# ------------------------------------------------------------------------------------------------ core builder
def _is_skipped(path: str) -> bool:
    return PurePosixPath(path).parts[0] in SKIP_TOP_DIRS if PurePosixPath(path).parts else True


def _find_inherited(index: dict[str, set[str]], directories: list[str], entities: dict[str, str],
                    suffix: str, exts: tuple[str, ...]) -> str | None:
    """BIDS inheritance: the most specific file ``<entities subset>_<suffix>.<ext>`` in the recording's directory
    or in a parent directory (subject, session, dataset root)."""
    best: tuple[int, int, str] | None = None
    best_path: str | None = None
    for depth, d in enumerate(directories):             # directories are ordered most specific first
        for name in index.get(d, ()):
            info = parse_bids_name(name)
            if not info or info["suffix"] != suffix or info["ext"] not in exts:
                continue
            ents = info["entities"]
            if all(entities.get(k) == v for k, v in ents.items()):
                score = (len(ents), -depth, name)
                if best is None or score > best:
                    best = (len(ents), -depth, name)
                    best_path = f"{d}/{name}" if d else name
        if best is not None:
            break                                       # a more specific level always wins
    return best_path


def discover_from_paths(files: Iterable[tuple[str, int | None]]) -> tuple[list[RecordingSpec], dict[str, int]]:
    """Find every EEG recording in a list of ``(relative POSIX path, size)`` and attach companions and sidecars."""
    files = [(p.lstrip("/"), s) for p, s in files]
    sizes = {p: s for p, s in files}
    index: dict[str, set[str]] = {}
    modalities: Counter = Counter()
    for p, _ in files:
        pp = PurePosixPath(p)
        index.setdefault(str(pp.parent) if str(pp.parent) != "." else "", set()).add(pp.name)
        if pp.parts[0].startswith("sub-") and len(pp.parts) >= 3 and not _is_skipped(p):
            modality = pp.parts[-2]
            if modality in ("eeg", "meg", "ieeg", "anat", "func", "dwi", "fmap", "beh", "pet", "nirs", "perf", "motion"):
                modalities[modality] += 1

    specs: list[RecordingSpec] = []
    for p, size in files:
        pp = PurePosixPath(p)
        if _is_skipped(p) or not pp.parts[0].startswith("sub-") or pp.parent.name != "eeg":
            continue
        info = parse_bids_name(pp.name)
        if not info or info["suffix"] != "eeg" or info["ext"] not in SUPPORTED_EXTENSIONS:
            continue
        ents = info["entities"]
        if "sub" not in ents:
            continue
        d = str(pp.parent)
        companions = []
        for ext in COMPANION_EXTENSIONS.get(info["ext"], ()):
            cand = f"{d}/{info['stem']}{ext}"
            if cand in sizes:
                companions.append(cand)
        subj_dir = pp.parts[0]
        dirs = [d]
        if len(pp.parts) >= 4:
            dirs.append(f"{pp.parts[0]}/{pp.parts[1]}")
        dirs += [subj_dir, ""]
        sidecars: dict[str, str] = {}
        js = _find_inherited(index, dirs, ents, "eeg", (".json",))
        ch = _find_inherited(index, dirs, ents, "channels", (".tsv",))
        ev = _find_inherited(index, dirs, ents, "events", (".tsv",))
        for k, v in (("json", js), ("channels", ch), ("events", ev)):
            if v:
                sidecars[k] = v
        specs.append(RecordingSpec(
            subject=f"sub-{ents['sub']}", session=f"ses-{ents['ses']}" if "ses" in ents else None,
            task=ents.get("task"), acq=ents.get("acq"), run=ents.get("run"), fmt=info["ext"].lstrip("."),
            path=p, size=size, companions=companions, sidecars=sidecars, entities=ents))
    specs.sort(key=lambda r: (natural_key(r.subject), natural_key(r.session or ""), natural_key(r.task or ""),
                              natural_key(r.acq or ""), natural_key(r.run or ""), r.path))
    return specs, dict(modalities)


def parse_participants(text: str) -> dict[str, dict[str, str]]:
    """``participants.tsv`` -> ``{"sub-001": {column: value}}`` (tolerates BOM and missing file)."""
    out: dict[str, dict[str, str]] = {}
    if not text:
        return out
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter="\t")
    for row in reader:
        pid = (row.get("participant_id") or "").strip()
        if pid:
            out[pid if pid.startswith("sub-") else f"sub-{pid}"] = {k: (v or "").strip() for k, v in row.items() if k}
    return out


def demographics(row: dict[str, str] | None) -> dict[str, str]:
    """Age, sex and handedness from a participants row, tolerant to column names (age/Age, sex/gender/Sex ...)."""
    row = {k.lower(): v for k, v in (row or {}).items()}
    pick = lambda *keys: next((row[k] for k in keys if k in row and row[k] not in ("", "n/a", "N/A")), "")  # noqa: E731
    return {"sex": pick("sex", "gender"), "age": pick("age"), "handedness": pick("handedness", "hand")}


# ------------------------------------------------------------------------------------------------ local BIDS
def discover_local(bids_dir: str | Path) -> DatasetInfo:
    root = Path(bids_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetNotFoundError(f"BIDS directory not found: {root}")
    files: list[tuple[str, int | None]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if d not in SKIP_TOP_DIRS]
        for name in filenames:
            full = Path(dirpath) / name
            try:
                size = full.stat().st_size if full.exists() else None
            except OSError:
                size = None
            files.append(((rel_dir / name).as_posix() if rel_dir != Path(".") else name, size))
    specs, modalities = discover_from_paths(files)
    desc, participants = {}, {}
    dd, pt = root / "dataset_description.json", root / "participants.tsv"
    try:
        if dd.exists():
            desc = json.loads(dd.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    try:
        if pt.exists():
            participants = parse_participants(pt.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        pass
    info = DatasetInfo(source="local", dataset_id=root.name, key=f"local:{root.as_posix()}",
                       title=str(desc.get("Name", "")), description=desc, snapshot=None, recordings=specs,
                       participants=participants, bids_dir=root, discovery_method="directory walk",
                       modalities_seen=modalities)
    if not specs:
        raise NoEEGDataError(_no_eeg_message(root.name, modalities))
    return info


def _no_eeg_message(name: str, modalities: dict[str, int]) -> str:
    seen = ", ".join(f"{k} ({v} files)" for k, v in sorted(modalities.items())) or "no sub-*/<modality>/ folders"
    return (f"'{name}' contains no EEG recordings in a supported format "
            f"({', '.join(SUPPORTED_EXTENSIONS)}) under sub-*/[ses-*/]eeg/. Found: {seen}.")


# ------------------------------------------------------------------------------------------------ OpenNeuro
def _version_key(tag: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"[.\-_]", tag.lstrip("v"))]


def _github_tree(http: Http, ep: Endpoints, ds: str, ref: str) -> tuple[list[tuple[str, int | None]], bool]:
    """All blobs of the git tree at ``ref``. Returns (files, complete). Falls back to a directory-by-directory
    walk when the recursive listing is truncated by GitHub."""
    base = f"{ep.github_api}/repos/OpenNeuroDatasets/{ds}/git/trees"
    data = http.get_json(f"{base}/{quote(ref, safe='')}?recursive=1")
    if not data.get("truncated"):
        return [(e["path"], e.get("size")) for e in data.get("tree", []) if e.get("type") == "blob"], True

    # truncated: walk only what we need (sub-*/[ses-*/]eeg and the root files)
    def walk(sha: str, prefix: str, depth: int) -> list[tuple[str, int | None]]:
        out: list[tuple[str, int | None]] = []
        tree = http.get_json(f"{base}/{quote(sha, safe=':')}").get("tree", [])
        for e in tree:
            path = f"{prefix}{e['path']}"
            if e["type"] == "blob":
                out.append((path, e.get("size")))
            elif e["type"] == "tree":
                name = e["path"]
                if depth == 0 and (name.startswith("sub-")):
                    out += walk(e["sha"], f"{path}/", 1)
                elif depth == 1 and (name.startswith("ses-") or name == "eeg"):
                    out += walk(e["sha"], f"{path}/", 2)
                elif depth == 2 and name == "eeg":
                    out += walk(e["sha"], f"{path}/", 3)
        return out

    return walk(ref, "", 0), False


def _s3_listing(http: Http, ep: Endpoints, ds: str) -> list[tuple[str, int | None]]:
    """Every object under ``<ds>/`` of the public OpenNeuro bucket (ListObjectsV2, paginated)."""
    files: list[tuple[str, int | None]] = []
    token = None
    while True:
        url = f"{ep.s3_base}?list-type=2&prefix={quote(ds + '/', safe='')}&max-keys=1000"
        if token:
            url += f"&continuation-token={quote(token, safe='')}"
        root = ET.fromstring(http.get_bytes(url))
        ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        q = (lambda t: f"s:{t}") if ns else (lambda t: t)
        for c in root.findall(q("Contents"), ns):
            key = c.findtext(q("Key"), default="", namespaces=ns)
            size = c.findtext(q("Size"), default=None, namespaces=ns)
            if key.startswith(ds + "/") and not key.endswith("/"):
                files.append((key[len(ds) + 1:], int(size) if size is not None else None))
        if (root.findtext(q("IsTruncated"), default="false", namespaces=ns) or "false").lower() == "true":
            token = root.findtext(q("NextContinuationToken"), default=None, namespaces=ns)
            if not token:
                break
        else:
            break
    return files


def _small_file(http: Http, ep: Endpoints, ds: str, ref: str | None, rel: str) -> str:
    """Read a small metadata file: pinned GitHub raw first, then the S3 bucket. Empty string if unavailable."""
    urls = []
    if ref:
        urls.append(f"{ep.github_raw}/OpenNeuroDatasets/{ds}/{quote(ref, safe='')}/{quote(rel, safe='/')}")
    urls.append(f"{ep.s3_base}/{ds}/{quote(rel, safe='/')}")
    for u in urls:
        try:
            text = http.get_text(u)
        except HttpError:
            continue
        if "/annex/objects/" in text[:200]:          # a git-annex pointer, not the content
            continue
        return text
    return ""


def discover_openneuro(dataset_id: str, snapshot: str | None = None, http: Http | None = None,
                       endpoints: Endpoints | None = None) -> DatasetInfo:
    """Inspect an OpenNeuro dataset without downloading EEG data. Raises :class:`DatasetNotFoundError`,
    :class:`NoEEGDataError` or :class:`DiscoveryError` with a message that can be shown to the user as is."""
    ds = dataset_id.strip()
    if not OPENNEURO_ID_RE.match(ds):
        raise DatasetNotFoundError(f"'{dataset_id}' is not an OpenNeuro dataset id (expected e.g. ds005385)")
    ds = ds.lower()
    http, ep = http or Http(), endpoints or Endpoints.from_env()

    files: list[tuple[str, int | None]] | None = None
    method, ref, warnings = "", None, []
    tags: list[str] = []
    github_error: str | None = None
    try:
        raw_tags = http.get_json(f"{ep.github_api}/repos/OpenNeuroDatasets/{ds}/tags?per_page=100")
        tags = sorted((t["name"] for t in raw_tags), key=_version_key)
        if snapshot:
            if snapshot not in tags:
                raise DiscoveryError(f"snapshot '{snapshot}' not found for {ds}. Available: {', '.join(tags) or 'none'}")
            ref = snapshot
        else:
            ref = tags[-1] if tags else "HEAD"
        files, complete = _github_tree(http, ep, ds, ref)
        method = f"GitHub tree ({ref})" + ("" if complete else ", directory walk")
        if not tags:
            warnings.append("no snapshot tags found; the default branch was inspected")
    except HttpError as e:
        github_error = str(e)
    if files is None:
        if snapshot:
            raise DiscoveryError(f"cannot inspect snapshot '{snapshot}' of {ds}: {github_error}. "
                                 "Set GITHUB_TOKEN if GitHub rate-limited the request.")
        try:
            files = _s3_listing(http, ep, ds)
        except (HttpError, ET.ParseError) as e:
            raise DiscoveryError(f"cannot reach OpenNeuro for {ds} (GitHub: {github_error}; S3: {e})") from e
        if not files:
            raise DatasetNotFoundError(f"dataset '{ds}' was not found on OpenNeuro (GitHub: {github_error}).")
        method = "S3 listing (latest snapshot)"
        warnings.append("GitHub was unavailable; the S3 listing shows the newest snapshot, not a pinned version")

    specs, modalities = discover_from_paths(files)
    if not specs:
        raise NoEEGDataError(_no_eeg_message(ds, modalities))

    desc: dict[str, Any] = {}
    text = _small_file(http, ep, ds, ref, "dataset_description.json")
    if text:
        try:
            desc = json.loads(text)
        except json.JSONDecodeError:
            warnings.append("dataset_description.json could not be parsed")
    participants = parse_participants(_small_file(http, ep, ds, ref, "participants.tsv"))
    return DatasetInfo(source="openneuro", dataset_id=ds, key=f"openneuro:{ds}", title=str(desc.get("Name", "")),
                       description=desc, snapshot=ref if ref and ref != "HEAD" else None, recordings=specs,
                       participants=participants, discovery_method=method, modalities_seen=modalities,
                       warnings=warnings)


# ------------------------------------------------------------------------------------------------ selection
def _numeric_id(label: str) -> int | None:
    body = label[4:] if label.startswith("sub-") else label
    return int(body) if body.isdigit() else None


def parse_subject_tokens(tokens: list[str]) -> list[tuple]:
    """Tokens of ``--subjects``: ``28``, ``1 to 20``, ``1-20``, ``sub-001``, ``PD01``, ``1,5,9`` (mixable)."""
    flat: list[str] = []
    for t in tokens:
        flat += [x for x in re.split(r"[,\s]+", t.strip()) if x]
    out: list[tuple] = []
    i = 0
    while i < len(flat):
        t = flat[i]
        if i + 2 < len(flat) and flat[i + 1].lower() == "to":
            a, b = flat[i], flat[i + 2]
            if a.isdigit() and b.isdigit():
                out.append(("range", int(a), int(b)))
                i += 3
                continue
            raise SelectionError(f"invalid range '{a} to {b}': both ends must be numbers")
        m = re.fullmatch(r"(\d+)-(\d+)", t)
        if m:
            out.append(("range", int(m.group(1)), int(m.group(2))))
        else:
            out.append(("single", t))
        i += 1
    return out


def resolve_subjects(tokens: list[str], available: list[str]) -> list[str]:
    """Map the user's subject tokens to subject labels that exist. Unknown tokens are an error, never skipped."""
    by_num: dict[int, list[str]] = {}
    by_label = {a.lower(): a for a in available}
    for a in available:
        n = _numeric_id(a)
        if n is not None:
            by_num.setdefault(n, []).append(a)
    chosen: list[str] = []
    unknown: list[str] = []
    for tok in parse_subject_tokens(tokens):
        if tok[0] == "range":
            lo, hi = sorted((tok[1], tok[2]))
            hit = [lab for n, labs in by_num.items() if lo <= n <= hi for lab in labs]
            if not hit:
                unknown.append(f"{lo} to {hi}")
            chosen += hit
        else:
            t = tok[1]
            if t.isdigit() and int(t) in by_num:
                chosen += by_num[int(t)]
            elif t.lower() in by_label:
                chosen.append(by_label[t.lower()])
            elif f"sub-{t}".lower() in by_label:
                chosen.append(by_label[f"sub-{t}".lower()])
            else:
                unknown.append(t)
    if unknown:
        raise SelectionError(f"subject(s) not found in the dataset (or without EEG data): {', '.join(unknown)}")
    return sorted(set(chosen), key=natural_key)


def _norm_entity(values: list[str] | None, prefix: str) -> set[str] | None:
    if not values:
        return None
    return {(v if v.lower().startswith(prefix + "-") else f"{prefix}-{v}").lower() for v in values}


def select_recordings(info: DatasetInfo, subjects: list[str] | None = None, sessions: list[str] | None = None,
                      tasks: list[str] | None = None, limit: int | None = None) -> dict[str, list[RecordingSpec]]:
    """Apply the user's filters. Returns ``{subject: [recordings]}`` in natural order; ``limit`` keeps the first N
    subjects after all other filters."""
    chosen = set(resolve_subjects(subjects, info.subjects)) if subjects else None
    ses_f, task_f = _norm_entity(sessions, "ses"), _norm_entity(tasks, "task")
    out: dict[str, list[RecordingSpec]] = {}
    for r in info.recordings:
        if chosen is not None and r.subject not in chosen:
            continue
        if ses_f is not None and (r.session is None or r.session.lower() not in ses_f):
            continue
        if task_f is not None and (r.task is None or f"task-{r.task}".lower() not in task_f):
            continue
        out.setdefault(r.subject, []).append(r)
    ordered = {s: out[s] for s in sorted(out, key=natural_key)}
    if limit is not None:
        if limit < 1:
            raise SelectionError("--limit must be at least 1")
        ordered = dict(list(ordered.items())[:limit])
    if not ordered:
        raise SelectionError("no recordings match the requested subjects / sessions / tasks. "
                             f"Available sessions: {', '.join(info.sessions) or 'none (single session)'}; "
                             f"tasks: {', '.join(info.tasks) or 'none'}")
    return ordered
