"""Batch orchestration: stream -> screen -> persist -> delete, resume, and subject-level assembly.

On-disk state (everything the resume logic and the global Excel need)::

    <output>/.openneuro_eeg_triage.json      marker: which dataset this directory belongs to
    <output>/audit_trail.jsonl               append-only event log
    <output>/sub-XXX/status.json             subject status + final per-recording results
    <output>/sub-XXX/recordings/<rec>.json   cached per-recording QC result (layers 1-3, spectrum, channel table)
    <output>/sub-XXX/recordings/<rec>.psd.npz  posterior-ROI spectrum (for the figures)
    <output>/sub-XXX/figures/*.png

The global workbook (``qc_summary.xlsx``) and the CSV/JSON summaries are always rebuilt from *all* ``sub-*``
folders, so a second run on other subjects extends and re-ranks the earlier results.
"""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import QC_LOGIC_VERSION, QCConfig
from .discovery import DatasetInfo, RecordingSpec, demographics
from .engine import (analyse_recording, check_montage, eeg_names_from_channels_tsv, finalize_records,
                     infer_condition, load_recording, metric, montage_layer, subject_label)
from .streamer import Backend, DownloadError
from .utils import (Logger, human_bytes, human_seconds, natural_key, now_iso, read_json, sha256_file, to_jsonable,
                    write_json_atomic)

MARKER = ".openneuro_eeg_triage.json"
GLOBAL_FILES = ("qc_summary.xlsx", "qc_results.csv", "qc_subjects.csv", "qc_summary.json", "audit_trail.jsonl")


class OutputDirError(Exception):
    """The output directory belongs to another dataset (or cannot be used)."""


@dataclass
class RunOptions:
    output_dir: Path
    work_dir: Path
    repo_dir: Path
    cleanup: bool = True
    plots: bool = True
    resume: bool = True
    force: bool = False
    overwrite_all: bool = False
    hash_inputs: bool = True
    stop_after_failures: int = 3
    summary_every: int = 25
    debug: bool = False
    download_method: str = "auto"
    snapshot: str | None = None
    command: str = ""


@dataclass
class BatchResult:
    subjects_requested: int = 0
    completed: int = 0
    skipped: int = 0
    partial: int = 0
    failed: int = 0
    interrupted: bool = False
    stopped_early: bool = False
    verdicts: Counter = field(default_factory=Counter)
    bytes_downloaded: int = 0
    bytes_deleted: int = 0
    cleanup_errors: int = 0
    elapsed_s: float = 0.0
    messages: list[str] = field(default_factory=list)
    unfinished: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if (self.interrupted or self.stopped_early or self.partial or self.failed) else 0


# ================================================================================ paths & audit
def sub_dir(out: Path, sub: str) -> Path:
    return Path(out) / sub


def status_path(out: Path, sub: str) -> Path:
    return sub_dir(out, sub) / "status.json"


def rec_dir(out: Path, sub: str) -> Path:
    return sub_dir(out, sub) / "recordings"


class AuditTrail:
    """Append-only ``audit_trail.jsonl``: one JSON object per event, written (and flushed) immediately."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        line = json.dumps(to_jsonable({"ts": now_iso(), "event": event, **fields}), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def prepare_output_dir(opts: RunOptions, info: DatasetInfo, backend_info: dict[str, Any], log: Logger) -> None:
    """Create the output directory, check that it belongs to this dataset, honour ``--overwrite-all``."""
    out = opts.output_dir
    out.mkdir(parents=True, exist_ok=True)
    marker = out / MARKER
    if marker.exists():
        try:
            old = read_json(marker)
        except Exception:
            old = {}
        if old.get("dataset_key") and old["dataset_key"] != info.key:
            if not opts.overwrite_all:
                raise OutputDirError(
                    f"output directory '{out}' already holds results for {old.get('dataset_key')}, "
                    f"but this run is for {info.key}. Use a different --output-dir, or --overwrite-all to discard "
                    "the old results.")
    if opts.overwrite_all:
        removed = 0
        for d in out.glob("sub-*"):
            if d.is_dir() and (d / "status.json").exists():
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        for name in GLOBAL_FILES:
            (out / name).unlink(missing_ok=True)
        if removed:
            log.info(f"  --overwrite-all: removed the previous results of {removed} subject(s).")
    write_json_atomic(marker, {
        "tool": "openneuro-eeg-triage", "version": __version__, "dataset_key": info.key, "dataset_id": info.dataset_id,
        "title": info.title, "snapshot": info.snapshot, "backend": backend_info, "updated_at": now_iso()})


# ================================================================================ one recording
def read_sidecar(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None or not Path(path).exists():
        return None, None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return (data, None) if isinstance(data, dict) else (None, "JSON sidecar is not an object")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _base_record(spec: RecordingSpec, cfg: QCConfig) -> dict[str, Any]:
    cond, src = infer_condition(spec.task, spec.acq, cfg)
    pair = [spec.session or "", "*" if src == "task" else (spec.task or ""), "*" if src == "acq" else (spec.acq or ""),
            spec.run or ""]
    return {"rec_id": spec.rec_id, "subject": spec.subject, "session": spec.session or "", "task": spec.task or "",
            "acq": spec.acq or "", "run": spec.run or "", "format": spec.fmt, "path": spec.path, "condition": cond,
            "cond_source": src, "pair_key": pair, "signature": cfg.fingerprint(), "qc_version": QC_LOGIC_VERSION,
            "analysed_at": now_iso(), "error": None, "error_code": None, "layers": {}, "occ": None, "ch_table": None,
            "psd_file": None, "inputs": [], "analysis_s": 0.0, "n_eeg": None, "line": {}, "clipped_frac": 0.0,
            "info": {}, "summary": {"ch_names": [], "sfreq": None}, "warnings": []}


def persist_record(out: Path, record: dict[str, Any], psd_occ: dict[str, Any] | None) -> None:
    d = rec_dir(out, record["subject"])
    d.mkdir(parents=True, exist_ok=True)
    if psd_occ is not None and len(psd_occ["channels"]):
        np.savez_compressed(d / f"{record['rec_id']}.psd.npz", freqs=psd_occ["freqs"], psd=psd_occ["psd"],
                            channels=np.array(psd_occ["channels"]))
        record["psd_file"] = f"{record['rec_id']}.psd.npz"
    write_json_atomic(d / f"{record['rec_id']}.json", record)


def load_cached_record(out: Path, sub: str, rec_id: str, cfg: QCConfig) -> dict[str, Any] | None:
    p = rec_dir(out, sub) / f"{rec_id}.json"
    if not p.exists():
        return None
    try:
        rec = read_json(p)
    except Exception:
        return None
    return rec if rec.get("signature") == cfg.fingerprint() else None


def screen_recording(spec: RecordingSpec, backend: Backend, cfg: QCConfig, opts: RunOptions, audit: AuditTrail,
                     log: Logger) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch, screen, persist and delete ONE recording. Returns ``(record, cleanup_info)``.

    The ``finally`` block deletes everything that was fetched for this recording, also after an error, but
    only after the metrics have been persisted (or the failure has been recorded).
    """
    out = opts.output_dir
    record = _base_record(spec, cfg)
    session = backend.new_session()
    cleanup_info: dict[str, Any] = {"deleted": 0, "bytes": 0, "errors": [], "kept": True}
    t0 = time.time()
    audit.log("recording_start", subject=spec.subject, recording=spec.rec_id, format=spec.fmt)
    try:
        try:
            backend.fetch_sidecars(spec, session)
            sidecar, sidecar_error = read_sidecar(session.sidecars.get("json"))
            tsv = eeg_names_from_channels_tsv(session.sidecars.get("channels"))
            tsv_types = tsv[1] if tsv else None
            if tsv:                                   # reject a non-standard cap BEFORE the big file is downloaded
                check = check_montage(tsv[0], cfg)
                if not check.supported:
                    layer = montage_layer(check, len(tsv[0]))
                    record.update(status="UNSUPPORTED", layers={"header": layer}, n_eeg=len(tsv[0]),
                                  summary={"ch_names": list(tsv[0]), "sfreq": None})
                    record["warnings"] = ["rejected from channels.tsv before downloading the recording"]
                    record["analysis_s"] = round(time.time() - t0, 2)
                    persist_record(out, record, None)
                    audit.log("screened", subject=spec.subject, recording=spec.rec_id, status="UNSUPPORTED",
                              downloaded=False)
                    return record, cleanup_info
            backend.fetch_data(spec, session)
            audit.log("downloaded", subject=spec.subject, recording=spec.rec_id, bytes=session.bytes_downloaded,
                      files=[p.name for p in session.files])
            if opts.hash_inputs:
                record["inputs"] = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                                    for p in session.files if p.exists() and p.is_file()
                                    and p not in session.sidecars.values()]
            file_kb = session.main.stat().st_size / 1024 if session.main is not None else None
            rec = load_recording(session.main, spec.fmt, cfg, tsv_types)       # type: ignore[arg-type]
            try:
                res = analyse_recording(rec, cfg, sidecar, sidecar_error, session.sidecars.get("events"), file_kb)
            finally:
                del rec
            ch = res["ch_table"]
            record.update(status=res["status"], layers=res["layers"], occ=res["occ"], line=res["line"],
                          clipped_frac=res["clipped_frac"], n_eeg=res["n_eeg"], info=res["info"],
                          summary=res["summary"], montage=res["montage"],
                          ch_table=ch.round(4).to_dict("records") if ch is not None else None)
            record["warnings"] = list(session.warnings)
            record["analysis_s"] = round(time.time() - t0, 2)
            persist_record(out, record, res["psd_occ"])
            audit.log("screened", subject=spec.subject, recording=spec.rec_id, status=res["status"])
            return record, cleanup_info
        except DownloadError as e:
            record.update(status="MISSING", error=str(e), error_code="DOWNLOAD_FAILED")
            record["analysis_s"] = round(time.time() - t0, 2)
            audit.log("download_failed", subject=spec.subject, recording=spec.rec_id, error=str(e))
            return record, cleanup_info                                       # not persisted: retried on the next run
        except KeyboardInterrupt:
            raise
        except Exception as e:                                                # an unreadable file becomes a FAIL row
            if opts.debug:
                raise
            record.update(status="ERROR", error=f"{type(e).__name__}: {e}", error_code="FILE_NOT_READABLE")
            record["analysis_s"] = round(time.time() - t0, 2)
            persist_record(out, record, None)
            audit.log("screen_error", subject=spec.subject, recording=spec.rec_id, error=record["error"])
            return record, cleanup_info
    finally:
        if opts.cleanup and backend.deletes_data:
            cleanup_info.update(backend.cleanup(session))
            audit.log("cleaned", subject=spec.subject, recording=spec.rec_id, deleted=cleanup_info["deleted"],
                      bytes=cleanup_info["bytes"], errors=cleanup_info["errors"])
        record["_bytes_downloaded"] = session.bytes_downloaded


# ================================================================================ one subject
def summarize_subject(sub: str, final: list[dict[str, Any]], participant: dict[str, str], status: str,
                      cfg: QCConfig) -> dict[str, Any]:
    """The Subjects-sheet row of one subject (all numbers come from the final per-recording results)."""
    verdicts = [r["overall"] for r in final]
    labels = [r["label"] for r in final]
    L = lambda r, layer, name, d=np.nan: metric(r["layers"], layer, name, d)    # noqa: E731
    all_bad = [c for r in final for c in L(r, "integrity", "bad_channels", [])]
    crit_bad = [c for r in final for c in L(r, "integrity", "critical_bad_channels", [])]
    codes = sorted({c for r in final for c in r["codes"]})

    def med(vals):
        v = [float(x) for x in vals if isinstance(x, (int, float, np.floating)) and np.isfinite(x)]
        return round(float(np.median(v)), 3) if v else None

    usable = sum(1 for lab in labels if lab in ("USABLE", "USABLE_WITH_FLAGS", "USABLE_NEEDS_INTERPOLATION"))
    n_eeg = Counter(r["n_eeg"] for r in final if r.get("n_eeg")).most_common(1)
    sf = Counter(r["summary"].get("sfreq") for r in final if r.get("summary", {}).get("sfreq")).most_common(1)
    label = subject_label(labels)
    overall = "NO DATA"
    vv = [v for v in verdicts if v in ("PASS", "FLAG", "FAIL")]
    if vv:
        overall = max(vv, key={"PASS": 0, "FLAG": 1, "FAIL": 2}.__getitem__)
    text = f"{usable} of {len(final)} recordings usable" + (f"; label {label}" if label != "NO DATA" else "")
    return {
        "subject": sub, "status": status, "verdict": overall, "label": label, "summary": text,
        "n_recordings": len(final),
        "n_PASS": verdicts.count("PASS"), "n_FLAG": verdicts.count("FLAG"), "n_FAIL": verdicts.count("FAIL"),
        "n_MISSING": verdicts.count("MISSING"),
        "n_USABLE": labels.count("USABLE"), "n_USABLE_WITH_FLAGS": labels.count("USABLE_WITH_FLAGS"),
        "n_USABLE_NEEDS_INTERPOLATION": labels.count("USABLE_NEEDS_INTERPOLATION"), "n_REJECT": labels.count("REJECT"),
        "usable_recordings": usable,
        "sessions": ", ".join(sorted({r["session"] for r in final if r["session"]}, key=natural_key)) or "single",
        "tasks": ", ".join(sorted({r["task"] for r in final if r["task"]}, key=natural_key)),
        "sex": participant.get("sex", ""), "age": participant.get("age", ""), "handedness": participant.get("handedness", ""),
        "n_eeg_channels": n_eeg[0][0] if n_eeg else None, "sfreq_hz": sf[0][0] if sf else None,
        "bad_channels (recordings flagged)": ", ".join(f"{k} x{v}" for k, v in Counter(all_bad).most_common(6)),
        "critical_channels_bad": ", ".join(f"{k} x{v}" for k, v in Counter(crit_bad).most_common()),
        "reason_codes": "; ".join(codes),
        "median_alpha_ratio_EC_EO": med([L(r, "spectral", "alpha_ratio_EC_EO") for r in final]),
        "median_line_noise_db": med([L(r, "spectral", "line_db") for r in final]),
        "analysis_seconds": round(sum(r.get("analysis_s", 0) for r in final), 1),
    }


def _skip_ok(out: Path, sub: str, specs: list[RecordingSpec], cfg: QCConfig) -> bool:
    p = status_path(out, sub)
    if not p.exists():
        return False
    try:
        st = read_json(p)
    except Exception:
        return False
    if st.get("status") != "COMPLETED" or st.get("config_fingerprint") != cfg.fingerprint():
        return False
    have = {r["rec_id"] for r in st.get("records", []) if r.get("status") != "MISSING"}
    return all(s.rec_id in have for s in specs)


def process_subject(sub: str, specs: list[RecordingSpec], info: DatasetInfo, backend: Backend, cfg: QCConfig,
                    opts: RunOptions, audit: AuditTrail, log: Logger, result: BatchResult,
                    backend_info: dict[str, Any]) -> str:
    """Screen all selected recordings of a subject, merge with earlier results of the same subject, write
    ``status.json``. Returns ``COMPLETED``, ``PARTIAL``, ``NO_DATA`` or ``SKIPPED``."""
    out = opts.output_dir
    if opts.resume and not opts.force and _skip_ok(out, sub, specs, cfg):
        return "SKIPPED"
    raw: dict[str, dict[str, Any]] = {}
    transient: dict[str, dict[str, Any]] = {}
    cleanup_log = {"deleted": 0, "bytes": 0, "errors": 0}
    status = "ERROR"
    for k, spec in enumerate(specs, 1):
        cached = None if (opts.force or not opts.resume) else load_cached_record(out, sub, spec.rec_id, cfg)
        t0 = time.time()
        if cached is not None:
            raw[spec.rec_id] = cached
            log.step(f"[{k}/{len(specs)}] {spec.rec_id}: cached ({cached['status']})")
            continue
        rec, cl = screen_recording(spec, backend, cfg, opts, audit, log)
        result.bytes_downloaded += rec.pop("_bytes_downloaded", 0)
        cleanup_log["deleted"] += cl["deleted"]
        cleanup_log["bytes"] += cl["bytes"]
        cleanup_log["errors"] += len(cl["errors"])
        for e in cl["errors"]:
            log.warn(f"cleanup problem for {spec.rec_id}: {e}")
        (transient if rec["status"] == "MISSING" else raw)[spec.rec_id] = rec
        log.step(f"[{k}/{len(specs)}] {spec.rec_id}: {rec['status']}"
                 + (f" - {rec['error']}" if rec.get("error") else "")
                 + f"  ({time.time() - t0:.1f} s"
                 + (f", deleted {human_bytes(cl['bytes'])}" if cl["bytes"] else "") + ")")

    # merge with earlier results of this subject that are not part of this selection (cumulative semantics)
    selected_ids = {s.rec_id for s in specs}
    rd = rec_dir(out, sub)
    if rd.exists():
        for p in rd.glob("*.json"):
            rid = p.stem
            if rid not in selected_ids and rid not in raw:
                old = load_cached_record(out, sub, rid, cfg)
                if old is not None:
                    raw[rid] = old
    order = sorted({**raw, **transient}.values(), key=lambda r: (natural_key(r["session"]), natural_key(r["task"]),
                                                                    natural_key(r["acq"]), natural_key(r["run"])))
    final = finalize_records(order, cfg)
    participant = demographics(info.participants.get(sub))
    n_missing = sum(1 for r in final if r["status"] == "MISSING")
    status = "COMPLETED" if n_missing == 0 else ("PARTIAL" if len(final) > n_missing else "NO_DATA")
    summary = summarize_subject(sub, final, participant, status, cfg)
    compact = []
    for r in final:
        c = {k: v for k, v in r.items() if k not in ("ch_table",)}
        compact.append(c)
    write_json_atomic(status_path(out, sub), {
        "subject": sub, "status": status, "qc_version": QC_LOGIC_VERSION, "config_fingerprint": cfg.fingerprint(),
        "dataset_key": info.key, "dataset_id": info.dataset_id, "snapshot": info.snapshot,
        "backend": backend_info, "updated_at": now_iso(), "participant": participant,
        "selected_rec_ids": sorted(selected_ids), "records": compact, "summary": summary,
        "cleanup": {**cleanup_log, "data_deleted": bool(opts.cleanup and backend.deletes_data)},
        "warnings": [w for r in final for w in r.get("warnings", [])]})
    audit.log("subject_done", subject=sub, status=status, label=summary["label"], verdict=summary["verdict"])
    if opts.plots and status != "NO_DATA":
        try:
            from .reporter import plot_subject
            n = plot_subject(out, sub, final, cfg)
            log.step(f"{n} figure(s) saved")
        except Exception as e:                       # a failed figure never fails the screening
            if opts.debug:
                raise
            log.warn(f"figures for {sub} skipped: {type(e).__name__}: {e}")
    for r in final:
        result.verdicts[r["overall"]] += 1
    result.cleanup_errors += cleanup_log["errors"]
    result.bytes_deleted += cleanup_log["bytes"]
    return status


# ================================================================================ whole batch
def run_batch(info: DatasetInfo, selection: dict[str, list[RecordingSpec]], backend: Backend, cfg: QCConfig,
              opts: RunOptions, log: Logger) -> BatchResult:
    from .reporter import write_global_outputs
    t_start = time.time()
    result = BatchResult(subjects_requested=len(selection))
    backend_info = backend.prepare()
    if backend_info.get("warning"):
        log.warn(backend_info["warning"])
    prepare_output_dir(opts, info, backend_info, log)
    audit = AuditTrail(opts.output_dir / "audit_trail.jsonl")
    audit.log("run_start", version=__version__, qc_version=QC_LOGIC_VERSION, config_fingerprint=cfg.fingerprint(),
              dataset=info.dataset_id, snapshot=info.snapshot, backend=backend_info, command=opts.command,
              n_subjects=len(selection), n_recordings=sum(len(v) for v in selection.values()),
              cleanup=bool(opts.cleanup and backend.deletes_data))
    meta = {"dataset_id": info.dataset_id, "title": info.title, "snapshot": info.snapshot, "source": info.source,
            "backend": backend_info, "command": opts.command}
    consecutive_failures = 0
    names = list(selection)
    try:
        for i, sub in enumerate(names, 1):
            specs = selection[sub]
            t0 = time.time()
            head = f"[{i}/{len(names)}] {sub}  ({len(specs)} recording(s))"
            try:
                status = process_subject(sub, specs, info, backend, cfg, opts, audit, log, result, backend_info)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if opts.debug:
                    raise
                status = "ERROR"
                write_json_atomic(status_path(opts.output_dir, sub), {
                    "subject": sub, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "updated_at": now_iso(),
                    "qc_version": QC_LOGIC_VERSION, "config_fingerprint": cfg.fingerprint(), "dataset_key": info.key,
                    "records": [], "summary": {"subject": sub, "status": "ERROR", "verdict": "NO DATA", "label": "NO DATA",
                                               "summary": f"{type(e).__name__}: {e}"}})
                audit.log("subject_error", subject=sub, error=f"{type(e).__name__}: {e}")
                log.error(f"{sub}: {type(e).__name__}: {e}")
            if status == "SKIPPED":
                result.skipped += 1
                log.info(f"{head}  already analysed, skipped")
                continue
            if status == "COMPLETED":
                result.completed += 1
                consecutive_failures = 0
            elif status == "PARTIAL":
                result.partial += 1
                result.unfinished.append(sub)
                consecutive_failures = 0
            else:
                result.failed += 1
                result.unfinished.append(sub)
                consecutive_failures += 1 if status == "NO_DATA" else 0
            st = read_json(status_path(opts.output_dir, sub)).get("summary", {})
            log.info(f"{head}  {status}  {st.get('summary', '')}  "
                     f"[{st.get('n_PASS', 0)} PASS / {st.get('n_FLAG', 0)} FLAG / {st.get('n_FAIL', 0)} FAIL"
                     f" / {st.get('n_MISSING', 0)} MISSING]  ({human_seconds(time.time() - t0)})")
            if opts.summary_every and i % opts.summary_every == 0:
                write_global_outputs(opts.output_dir, cfg, meta, log, quiet=True)
            if opts.stop_after_failures and consecutive_failures >= opts.stop_after_failures:
                result.stopped_early = True
                result.messages.append(f"stopped after {consecutive_failures} subjects in a row without any usable "
                                       "download (network problem?). Run the same command again to continue.")
                log.warn(result.messages[-1])
                break
    except KeyboardInterrupt:
        result.interrupted = True
        result.messages.append("interrupted (Ctrl+C). Finished subjects are saved; run the same command to continue.")
        audit.log("interrupted")
    finally:
        result.elapsed_s = time.time() - t_start
        try:
            write_global_outputs(opts.output_dir, cfg, meta, log)
        except Exception as e:                       # the per-subject results are safe; the summary can be rebuilt
            if opts.debug:
                raise
            log.warn(f"could not write the global summary: {type(e).__name__}: {e}")
        audit.log("run_end", completed=result.completed, skipped=result.skipped, partial=result.partial,
                  failed=result.failed, interrupted=result.interrupted, elapsed_s=round(result.elapsed_s, 1))
    return result
