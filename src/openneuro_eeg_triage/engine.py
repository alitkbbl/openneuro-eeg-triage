"""The 5-layer QC engine.

Layer 1  Header & metadata     channel count, sampling rate, duration, names, standard montage, sidecar
Layer 2  Channel integrity     flat / clipped / noisy / spatially uncorrelated channels, frozen segments
Layer 3  Amplitude             typical amplitude on a baseline-corrected copy of the signal
Layer 4  Spectral              alpha peak, Eyes Closed / Eyes Open alpha blocking, mains line noise
Layer 5  Consistency           same channels, order and sampling rate across the recordings of a subject

Design rule (unchanged from the prototype): this is a *triage*. Only catastrophic, unrepairable recordings are
rejected (FAIL). Everything healthy or repairable is kept, with a short reason code for every finding.

Layers 1-3 and the PSD are computed per recording, in memory, while the file is on disk. Layers 4 (pairing) and
5 need several recordings and are assembled afterwards from small per-recording summaries
(:func:`finalize_records`), so the raw data can be deleted immediately after each recording.
"""
from __future__ import annotations

import copy
import gc
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.integrate import trapezoid

from .config import (AUX_NAME_PATTERN, AUX_TSV_TYPES, CORE_ANCHORS, LABEL_RANK, LEGACY_NAME_MAP,
                     VERDICT_RANK, QCConfig)
from .utils import nan_if_none

LAYER_ORDER = ["header", "integrity", "amplitude", "spectral", "consistency"]
LAYER_PREFIX = {"header": "hdr", "integrity": "int", "amplitude": "amp", "spectral": "spec", "consistency": "con"}
OVERVIEW_STEP = 25

#: Every reason code the engine can emit: code -> (layer, default verdict, meaning). Used by the Excel legend.
REASON_CODES: dict[str, tuple[str, str, str]] = {
    "UNSUPPORTED_MONTAGE": ("header", "FAIL", "Channel names are not a standard 10-20 / 10-10 / 10-05 system (e.g. EGI E1..E128, unmapped BioSemi A1..A32). Not screened."),
    "TOO_FEW_EEG_CHANNELS": ("header", "FAIL", "Fewer EEG channels than the configured minimum."),
    "SFREQ_TOO_LOW": ("header", "FAIL", "Sampling rate below the configured minimum."),
    "SFREQ_ABOVE_MAX": ("header", "FLAG", "Sampling rate above the configured maximum."),
    "DURATION_TOO_SHORT": ("header", "FAIL", "Recording shorter than the minimum duration."),
    "DURATION_OUTSIDE_RANGE": ("header", "FLAG", "Duration outside the expected range."),
    "CHANNEL_NAMES_INVALID": ("header", "FAIL", "Empty or duplicate channel names (after normalisation)."),
    "REQUIRED_CHANNELS_MISSING": ("header", "FAIL", "A required channel (O1 or O2) is missing."),
    "SIDECAR_SAMPLING_RATE_MISMATCH": ("header", "FLAG", "SamplingFrequency in the JSON sidecar differs from the file."),
    "SIDECAR_CHANNEL_COUNT_MISMATCH": ("header", "FLAG", "EEGChannelCount in the JSON sidecar differs from the file."),
    "SIDECAR_NOT_NUMERIC": ("header", "FLAG", "Non-numeric sidecar values."),
    "SIDECAR_UNREADABLE": ("header", "FLAG", "The JSON sidecar could not be parsed."),
    "FILE_TOO_SMALL": ("header", "FAIL", "Main file smaller than the minimum size (broken download or dangling link)."),
    "CRITICAL_CHANNELS_BAD": ("integrity", "FAIL", "Two or more critical channels are bad."),
    "CRITICAL_CHANNEL_BAD": ("integrity", "FLAG", "One critical channel is bad (can be interpolated)."),
    "TOO_MANY_BAD_CHANNELS": ("integrity", "FAIL", "More bad channels than can be interpolated reliably."),
    "USABLE_NEEDS_INTERPOLATION": ("integrity", "PASS", "Some bad channels; the recording is usable after interpolating them."),
    "LOW_GLOBAL_SPATIAL_CORRELATION": ("integrity", "FLAG", "Most channels fall below the correlation threshold: the reference is not trusted, nobody is marked bad."),
    "RECORDING_FROZEN": ("integrity", "FAIL", "All channels frozen for a large part of the recording."),
    "FROZEN_SEGMENT": ("integrity", "FLAG", "All channels frozen for a noticeable time."),
    "AMPLITUDE_OUT_OF_RANGE": ("amplitude", "FAIL", "Typical amplitude outside the physiological range (probable scale problem)."),
    "HIGH_AMPLITUDE_CHANNELS": ("amplitude", "FLAG", "Many channels above the amplitude limit."),
    "OCCIPITAL_CHANNELS_MISSING": ("spectral", "FAIL", "No occipital channel available for the spectral check."),
    "DEAD_SPECTRUM": ("spectral", "FAIL", "No signal power at all on the occipital channels."),
    "LINE_NOISE_SATURATION": ("spectral", "FAIL", "High mains noise together with clipping of at least half of the channels."),
    "HIGH_LINE_NOISE": ("spectral", "FLAG", "Mains noise far above its flanks (notch filter needed)."),
    "LINE_NOISE": ("spectral", "FLAG", "Mains noise above its flanks."),
    "LOW_ALPHA_PROMINENCE": ("spectral", "FLAG", "Weak alpha peak in an Eyes-Closed recording (common in healthy people)."),
    "NO_ALPHA_BLOCKING": ("spectral", "FLAG", "Alpha does not drop from Eyes Closed to Eyes Open on any channel."),
    "WEAK_ALPHA_BLOCKING": ("spectral", "FLAG", "Median Eyes-Closed/Eyes-Open alpha ratio below the limit."),
    "ALPHA_RATIO_NOT_COMPUTED": ("spectral", "FLAG", "The matching Eyes-Closed/Eyes-Open partner recording is missing or unusable."),
    "CHANNEL_LAYOUT_DIFFERS": ("consistency", "FAIL", "Channel names or order differ from the other recordings of the subject."),
    "SFREQ_DIFFERS": ("consistency", "FAIL", "Sampling rate differs from the other recordings of the subject."),
    "FILE_NOT_READABLE": ("load", "FAIL", "The file could not be read."),
    "RECORDING_MISSING": ("load", "MISSING", "The recording is not available."),
    "DOWNLOAD_FAILED": ("load", "MISSING", "The download failed or was incomplete. Retried on the next run."),
}

LABEL_TEXT = {
    "USABLE": "usable as it is",
    "USABLE_WITH_FLAGS": "usable; flags (for example line noise, weak alpha) are worth a look",
    "USABLE_NEEDS_INTERPOLATION": "usable after interpolating the bad channels",
    "REJECT": "not usable (FAIL)",
    "MISSING": "recording not available",
}
SUBJECT_LABELS = list(LABEL_RANK) + ["PARTIALLY_USABLE"]


# ================================================================================ result helpers
def worst_verdict(verdicts, empty: str = "PASS") -> str:
    """Worst verdict in a list (FAIL > FLAG > PASS). Other values (N/A, MISSING) are ignored."""
    valid = [v for v in verdicts if v in VERDICT_RANK]
    return max(valid, key=VERDICT_RANK.__getitem__) if valid else empty


def make_issue(verdict: str, metric: str, value: Any, threshold: Any, code: str | None = None,
               **extra: Any) -> dict[str, Any]:
    """One finding of a layer. ``code`` is the short machine-readable reason, e.g. HIGH_LINE_NOISE."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        value = round(value, 3)
    return {"verdict": verdict, "code": code or metric.upper(), "metric": metric, "value": value,
            "threshold": threshold, **extra}


def make_result(layer: str, metrics: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    return {"layer": layer, "verdict": worst_verdict((i["verdict"] for i in issues), empty="PASS"),
            "metrics": metrics, "issues": issues}


# ================================================================================ channel names & montage
@lru_cache(maxsize=1)
def standard_names() -> dict[str, str]:
    """Lower-case -> canonical name for every channel of the MNE 10-05 template (fiducials excluded)."""
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            montage = mne.channels.make_standard_montage("colin27_1005")     # MNE >= 1.13 name
        except Exception:
            montage = mne.channels.make_standard_montage("standard_1005")
    return {n.lower(): n for n in montage.ch_names if n not in ("Nz", "LPA", "RPA")}


@lru_cache(maxsize=1)
def standard_positions() -> dict[str, np.ndarray]:
    """3-D template positions (m) of the 10-05 channels, for the topomap figure only."""
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            montage = mne.channels.make_standard_montage("colin27_1005")
        except Exception:
            montage = mne.channels.make_standard_montage("standard_1005")
    pos = montage.get_positions()["ch_pos"]
    return {k: np.asarray(v, dtype=float) for k, v in pos.items() if k not in ("Nz", "LPA", "RPA")}


_PREFIX_RE = re.compile(r"^\s*EEG[\s_:.\-]*", re.I)
_SUFFIX_RE = re.compile(r"[\s_\-]+(REF|LE|RE|AV|AVG|AVE|AVERAGE|LINKED|CAR|A1A2|M1M2|A1|A2|M1|M2)$", re.I)
_AUX_RE = re.compile(AUX_NAME_PATTERN, re.I)


def _clean_label(name: str) -> str:
    """Strip hardware decoration: ``EEG `` prefix, reference suffixes (``-REF``, ``-LE`` ...), padding dots."""
    s = str(name).strip()
    stripped = _PREFIX_RE.sub("", s)
    if stripped:
        s = stripped
    for _ in range(2):
        s = _SUFFIX_RE.sub("", s)
    return s.strip(" .")


def normalize_channel_name(name: str) -> tuple[str, bool]:
    """Map a hardware channel label to its standard 10-20 / 10-10 / 10-05 name.

    ``EEG Fz-REF`` -> ``Fz``, ``fp1`` -> ``Fp1``, ``OZ`` -> ``Oz``, legacy ``T3`` -> ``T7`` (``T4->T8``,
    ``T5->P7``, ``T6->P8``). Returns ``(name, recognised)``. An unrecognised label is returned unchanged
    (only stripped), never guessed.
    """
    s = _clean_label(name)
    legacy = LEGACY_NAME_MAP.get(s.upper())
    if legacy:
        return legacy, True
    canonical = standard_names().get(s.lower())
    if canonical:
        return LEGACY_NAME_MAP.get(canonical.upper(), canonical), True
    return str(name).strip(), False


@dataclass
class MontageCheck:
    supported: bool
    n_channels: int
    n_recognized: int
    fraction: float
    anchors_found: list[str]
    unrecognized: list[str]
    mapped: list[str]
    legacy_mapped: list[str]
    reason: str = ""


def check_montage(raw_names: list[str], cfg: QCConfig) -> MontageCheck:
    """The standard-montage gatekeeper: is this a 10-20 / 10-10 / 10-05 cap?

    Supported when at least ``montage_min_standard_fraction`` of the EEG channels carry a standard name AND at
    least ``montage_min_anchors`` of the 10 core anchors (Fz Cz Pz Oz F3 F4 C3 C4 O1 O2) can be verified.
    """
    mapped, ok, legacy = [], [], []
    for n in raw_names:
        m, r = normalize_channel_name(n)
        mapped.append(m)
        ok.append(r)
        if r and LEGACY_NAME_MAP.get(_clean_label(n).upper()):
            legacy.append(f"{n}->{m}")
    present = {m for m, r in zip(mapped, ok) if r}
    anchors = [a for a in CORE_ANCHORS if a in present]
    frac = (sum(ok) / len(ok)) if ok else 0.0
    supported = bool(ok) and frac >= cfg.montage_min_standard_fraction and len(anchors) >= cfg.montage_min_anchors
    unrec = [n for n, r in zip(raw_names, ok) if not r]
    reason = ""
    if not supported:
        reason = (f"{sum(ok)} of {len(ok)} channels carry standard 10-20/10-10/10-05 names "
                  f"({frac:.0%}; need >= {cfg.montage_min_standard_fraction:.0%}), "
                  f"{len(anchors)} of {len(CORE_ANCHORS)} core anchors found (need >= {cfg.montage_min_anchors}); "
                  f"unrecognised e.g. {', '.join(unrec[:6])}")
    return MontageCheck(supported, len(ok), int(sum(ok)), float(frac), anchors, unrec, mapped, legacy, reason)


def montage_layer(check: MontageCheck, n_eeg: int) -> dict[str, Any]:
    """Header layer for a recording rejected on its channel names alone (no signal was read)."""
    issues = [make_issue("FAIL", "standard_montage", check.reason,
                         "channel labels must follow the standard 10-20 / 10-10 / 10-05 system",
                         code="UNSUPPORTED_MONTAGE")]
    metrics = {"n_eeg_channels": n_eeg, "standard_montage": False, "n_standard_names": check.n_recognized,
               "anchors_found": check.anchors_found, "unrecognised_names": check.unrecognized[:20]}
    return make_result("header", metrics, issues)


def eeg_names_from_channels_tsv(path: Path | None) -> tuple[list[str], dict[str, str]] | None:
    """EEG channel names (and a name -> type map) from a BIDS ``channels.tsv``; ``None`` if unusable."""
    if path is None or not Path(path).exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception:
        return None
    if "name" not in df.columns:
        return None
    types = {r["name"]: (r["type"].strip().upper() if "type" in df.columns else "") for _, r in df.iterrows()}
    if "type" in df.columns:
        eeg = [n for n, t in types.items() if t == "EEG"]
    else:
        eeg = [n for n in types if not _AUX_RE.match(n.strip())]
    return (eeg, types) if eeg else None


def infer_condition(task: str | None, acq: str | None, cfg: QCConfig) -> tuple[str, str | None]:
    """Eyes ``closed`` / ``open`` / ``unknown`` from the task or acq label (or the explicit CLI override).

    Returns ``(condition, source_entity)``; ``source_entity`` is the BIDS entity that carries the condition so
    that Eyes-Closed and Eyes-Open recordings can be paired on all the *other* entities.
    A bare ``rest`` stays ``unknown``: the eye state cannot be inferred, so no alpha FLAG is raised for it.
    """
    forced_c = {t.lower() for t in cfg.closed_tasks}
    forced_o = {t.lower() for t in cfg.open_tasks}
    for source, value in (("task", task), ("acq", acq)):
        if not value:
            continue
        low = value.lower()
        if low in forced_c:
            return "closed", source
        if low in forced_o:
            return "open", source
    for source, value in (("task", task), ("acq", acq)):
        if not value:
            continue
        low = value.lower()
        tokens = {t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", value)} | {low}
        closed = bool(tokens & {"ec", "closed", "eyesclosed", "eyeclosed"}) or "closed" in low
        opened = bool(tokens & {"eo", "open", "eyesopen", "eyeopen"}) or "open" in low
        if closed and not opened:
            return "closed", source
        if opened and not closed:
            return "open", source
    return "unknown", None


# ================================================================================ recording container
@dataclass
class Recording:
    """One recording in memory. Signals in microvolts. ``ch_names`` are the *normalised* EEG channel names."""
    ch_names: list[str]
    raw_ch_names: list[str]
    sfreq: float
    data_uv: np.ndarray
    annotations: list[dict[str, Any]] = field(default_factory=list)
    extra_channels: list[tuple[str, str]] = field(default_factory=list)
    stim_data: dict[str, np.ndarray] = field(default_factory=dict)
    cfg: QCConfig = field(default_factory=QCConfig)
    _dt: np.ndarray | None = field(default=None, repr=False)

    @property
    def data_dt(self) -> np.ndarray:
        """Baseline-corrected copy (median removed + 0.5 Hz high-pass), computed once, in RAM only."""
        if self._dt is None:
            self._dt = detrend_signal(self.data_uv, self.sfreq, self.cfg)
        return self._dt

    def release_dt(self) -> None:
        self._dt = None

    @property
    def n_times(self) -> int:
        return int(self.data_uv.shape[1])

    @property
    def n_eeg(self) -> int:
        return int(self.data_uv.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_times / self.sfreq


def read_raw_any(path: Path, fmt: str):
    """Read an EEG file with MNE (EDF, BDF, BrainVision, EEGLAB, FIF), preloaded."""
    import mne
    fmt = fmt.lower().lstrip(".")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        if fmt == "edf":
            return mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        if fmt == "bdf":
            return mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
        if fmt == "vhdr":
            return mne.io.read_raw_brainvision(str(path), preload=True, verbose="ERROR")
        if fmt == "set":
            return mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
        if fmt == "fif":
            return mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR")
    raise ValueError(f"unsupported EEG format: {fmt!r}")


def recording_from_raw(raw, cfg: QCConfig, tsv_types: dict[str, str] | None = None) -> Recording:
    """Build a :class:`Recording` from an MNE Raw. Only scalp-EEG channels are analysed."""
    tsv_types = tsv_types or {}
    names, types = list(raw.ch_names), raw.get_channel_types()
    eeg_idx, aux = [], []
    for i, (n, t) in enumerate(zip(names, types)):
        tsv_t = tsv_types.get(n, "").upper()
        is_aux = (t != "eeg") or (tsv_t in AUX_TSV_TYPES) or (not tsv_t and bool(_AUX_RE.match(n.strip())))
        (aux if is_aux else eeg_idx).append(i)
    if not eeg_idx:
        raise ValueError("the file contains no EEG channels")
    data = raw.get_data(picks=eeg_idx) * 1e6
    stim = {names[i]: raw.get_data(picks=[i])[0] for i in aux if types[i] == "stim"}
    ann = [{"onset": float(a["onset"]), "duration": float(a["duration"]), "description": str(a["description"])}
           for a in raw.annotations]
    raw_names = [names[i] for i in eeg_idx]
    return Recording(ch_names=[normalize_channel_name(n)[0] for n in raw_names], raw_ch_names=raw_names,
                     sfreq=float(raw.info["sfreq"]), data_uv=data, annotations=ann,
                     extra_channels=[(names[i], types[i]) for i in aux], stim_data=stim, cfg=cfg)


def load_recording(path: Path, fmt: str, cfg: QCConfig, tsv_types: dict[str, str] | None = None) -> Recording:
    raw = read_raw_any(path, fmt)
    try:
        return recording_from_raw(raw, cfg, tsv_types)
    finally:
        del raw
        gc.collect()


# ================================================================================ signal helpers
def _norm_name(name: str) -> str:
    s = name.lower().strip()
    if s.startswith("eeg"):
        s = s[3:]
    s = "".join(ch for ch in s if ch.isalnum())
    return s[:-3] if s.endswith("ref") else s


def find_channel(ch_names: list[str], target: str) -> int | None:
    """Index of a channel, case-insensitive; exact name first, then the cleaned-up name."""
    lowered = [c.lower() for c in ch_names]
    if target.lower() in lowered:
        return lowered.index(target.lower())
    wanted = _norm_name(target)
    for i, c in enumerate(ch_names):
        if _norm_name(c) == wanted:
            return i
    return None


def detrend_signal(x: np.ndarray, fs: float, cfg: QCConfig) -> np.ndarray:
    """Baseline-corrected COPY of the signal (RAM only; the raw data are never modified)."""
    method = cfg.detrend_method
    if method == "none":
        return x
    xm = x - np.median(x, axis=1, keepdims=True)
    if method == "linear":
        return signal.detrend(xm, axis=1, type="linear")
    if method != "highpass":
        raise ValueError(f"Unknown detrend_method: {method!r}")
    sos = signal.butter(cfg.detrend_hp_order, cfg.detrend_hp_hz, btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, xm, axis=1)


def window_std_uv(x: np.ndarray, fs: float, win_s: float = 1.0) -> np.ndarray:
    """Typical amplitude per channel: median over 1-s windows of the STD after removing a line in each window."""
    w = int(round(win_s * fs))
    n_win = x.shape[1] // w
    if w < 2 or n_win < 1:
        return x.std(axis=1)
    seg = x[:, : n_win * w].reshape(x.shape[0], n_win, w)
    seg = signal.detrend(seg, axis=-1, type="linear")
    return np.median(seg.std(axis=-1), axis=1)


def longest_equal_run(x: np.ndarray) -> int:
    change = np.flatnonzero(np.diff(x) != 0)
    edges = np.concatenate(([-1], change, [len(x) - 1]))
    return int(np.diff(edges).max())


def find_frozen(x: np.ndarray, fs: float, cfg: QCConfig) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Segments in which ALL channels are frozen at the same time (a recording-level dropout, not a bad electrode)."""
    min_samples = max(2, int(round(cfg.frozen_min_s * fs)))
    same = np.all(np.diff(x, axis=1) == 0, axis=0)
    pad = np.concatenate(([False], same, [False]))
    edges = np.flatnonzero(pad[1:] != pad[:-1])
    mask = np.zeros(x.shape[1], dtype=bool)
    segments = []
    for s0, e0 in zip(edges[::2], edges[1::2]):
        length = e0 - s0 + 1
        if length >= min_samples:
            mask[s0:s0 + length] = True
            segments.append((float(s0 / fs), float(length / fs)))
    return mask, segments


# ================================================================================ LAYER 1: header & metadata
def header_check(rec: Recording, cfg: QCConfig, montage: MontageCheck, sidecar: dict | None = None,
                 sidecar_error: str | None = None, file_kb: float | None = None) -> dict[str, Any]:
    """Layer 1: channel count, sampling rate, duration, channel names, standard montage, required channels, sidecar."""
    issues: list[dict[str, Any]] = []
    names, dur = rec.ch_names, rec.duration_s
    aux = rec.extra_channels or []
    n_empty = sum(1 for n in names if not n.strip())
    duplicates = sorted(n for n, k in Counter(names).items() if k > 1)
    hints = tuple(h.upper() for h in cfg.non_eeg_name_hints)
    non_eeg_like = [n for n in names + [a[0] for a in aux] if any(h in n.upper() for h in hints)]

    if file_kb is not None and file_kb < cfg.min_file_kb:
        issues.append(make_issue("FAIL", "file_size_kb", file_kb, f">= {cfg.min_file_kb:g} kB", code="FILE_TOO_SMALL"))
    if not montage.supported:
        issues.append(make_issue("FAIL", "standard_montage", montage.reason,
                                 "channel labels must follow the standard 10-20 / 10-10 / 10-05 system",
                                 code="UNSUPPORTED_MONTAGE"))
    if len(names) < cfg.min_eeg_channels:
        issues.append(make_issue("FAIL", "n_eeg_channels", len(names), f">= {cfg.min_eeg_channels}", code="TOO_FEW_EEG_CHANNELS"))
    if rec.sfreq < cfg.min_sfreq_hz - 1e-9:
        issues.append(make_issue("FAIL", "sfreq_hz", rec.sfreq, f">= {cfg.min_sfreq_hz:g} Hz", code="SFREQ_TOO_LOW"))
    elif cfg.max_sfreq_hz is not None and rec.sfreq > cfg.max_sfreq_hz + 1e-9:
        issues.append(make_issue("FLAG", "sfreq_hz", rec.sfreq, f"<= {cfg.max_sfreq_hz:g} Hz", code="SFREQ_ABOVE_MAX"))
    lo, hi = cfg.duration_ok_range_s
    if dur < cfg.duration_fail_below_s:
        issues.append(make_issue("FAIL", "duration_s", dur, f">= {cfg.duration_fail_below_s:.0f} s", code="DURATION_TOO_SHORT"))
    elif not (lo <= dur <= hi):
        issues.append(make_issue("FLAG", "duration_s", dur, f"between {lo:.0f} and {hi:.0f} s", code="DURATION_OUTSIDE_RANGE"))
    if n_empty or duplicates:
        issues.append(make_issue("FAIL", "channel_names_unique_and_not_empty",
                                 f"{n_empty} empty, duplicates: {duplicates}", "all names unique and not empty",
                                 code="CHANNEL_NAMES_INVALID"))
    required = {}
    for target in cfg.required_channels:
        idx = find_channel(names, target)
        required[target] = names[idx] if idx is not None else None
    missing = [t for t, v in required.items() if v is None]
    if missing and montage.supported:
        issues.append(make_issue("FAIL", "required_channels_missing", ", ".join(missing),
                                 f"{', '.join(cfg.required_channels)} must exist", code="REQUIRED_CHANNELS_MISSING"))

    json_sf = json_n = None
    if sidecar is not None:
        json_sf, json_n = sidecar.get("SamplingFrequency"), sidecar.get("EEGChannelCount")
        try:
            if json_sf is not None and abs(float(json_sf) - rec.sfreq) > 1e-6:
                issues.append(make_issue("FLAG", "sidecar_SamplingFrequency", json_sf, f"equal to the file value {rec.sfreq}",
                                         code="SIDECAR_SAMPLING_RATE_MISMATCH"))
            if json_n is not None and int(json_n) != len(names):
                issues.append(make_issue("FLAG", "sidecar_EEGChannelCount", json_n, f"equal to the file value {len(names)}",
                                         code="SIDECAR_CHANNEL_COUNT_MISMATCH"))
        except (TypeError, ValueError):
            issues.append(make_issue("FLAG", "sidecar_values_not_numeric", f"{json_sf}, {json_n}", "numbers",
                                     code="SIDECAR_NOT_NUMERIC"))
    elif sidecar_error:
        issues.append(make_issue("FLAG", "sidecar_unreadable", sidecar_error, "valid JSON", code="SIDECAR_UNREADABLE"))

    roi = [c for c in cfg.psd_channels if find_channel(names, c) is not None]
    metrics = {
        "n_eeg_channels": len(names), "aux_channels": [f"{n} ({t})" for n, t in aux], "sfreq_hz": rec.sfreq,
        "duration_s": round(dur, 2), "n_samples": rec.n_times, "non_eeg_like_names": non_eeg_like,
        "standard_montage": montage.supported, "n_standard_names": montage.n_recognized,
        "anchors_found": montage.anchors_found, "legacy_names_mapped": montage.legacy_mapped,
        "unrecognised_names": montage.unrecognized[:20], "posterior_roi": roi,
        "required_channels_found": required, "sidecar_found": sidecar is not None,
        "sidecar_SamplingFrequency": json_sf, "sidecar_EEGChannelCount": json_n, "channel_names": names,
    }
    return make_result("header", metrics, issues)


# ================================================================================ LAYER 2: channel integrity
def corr_thresholds(ch_names: list[str], cfg: QCConfig) -> np.ndarray:
    """Per-channel spatial-correlation threshold: relaxed for the occipital channels, scaled for sparse caps."""
    relaxed = {_norm_name(k) for k in cfg.corr_relaxed_channels}
    scale = cfg.corr_scale(len(ch_names))
    return np.array([(cfg.corr_threshold_relaxed if _norm_name(n) in relaxed else cfg.corr_threshold) * scale
                     for n in ch_names], dtype=float)


def spatial_correlation(rec: Recording, cfg: QCConfig) -> dict[str, Any]:
    """|Pearson r| of every channel with the MEDIAN of all channels in 5-s windows (band 1-45 Hz).

    A healthy channel shares part of its signal with the others (volume conduction, reference); a noisy or
    disconnected channel that is neither flat nor clipped does not. The median is the reference because one
    bad channel cannot move it.
    """
    fs = rec.sfreq
    q = max(1, int(round(fs / cfg.corr_target_fs_hz)))
    lo, hi = cfg.corr_band_hz
    hi = min(cfg.band_hi(hi, fs), 0.45 * fs)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y_full = signal.sosfiltfilt(sos, rec.data_dt, axis=1)
    y = np.ascontiguousarray(y_full[:, ::q])
    del y_full
    fs_y = fs / q
    n_ch, n = y.shape
    w = int(round(cfg.corr_window_s * fs_y))
    n_win = n // w
    if n_win < 1:
        w, n_win = n, 1
    seg = y[:, : n_win * w].reshape(n_ch, n_win, w)
    ref = np.median(seg, axis=0)
    sc = seg - seg.mean(axis=-1, keepdims=True)
    rc = ref - ref.mean(axis=-1, keepdims=True)
    num = (sc * rc[None]).sum(axis=-1)
    den = np.sqrt((sc ** 2).sum(axis=-1)) * np.sqrt((rc ** 2).sum(axis=-1))[None]
    r = np.abs(np.divide(num, den, out=np.zeros_like(num), where=den > 0))
    thr = corr_thresholds(rec.ch_names, cfg)
    return {"median": np.median(r, axis=1), "low_frac": (r < thr[:, None]).mean(axis=1),
            "windows": r.astype(np.float32), "window_s": w / fs_y, "thresholds": thr}


def compute_channel_table(rec: Recording, cfg: QCConfig, corr: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-channel statistics in uV, one row per EEG channel.

    RAW signal: flat, clipping, frozen samples, DC offset, drift. BASELINE-CORRECTED copy: amplitude, variance,
    skewness, kurtosis, percentiles, spatial correlation.
    """
    x, xd = rec.data_uv, rec.data_dt
    n_ch, n = x.shape
    p_lo, p_hi = np.percentile(xd, [0.5, 99.5], axis=1)
    med = np.median(x, axis=1)
    frozen, _ = find_frozen(x, rec.sfreq, cfg)
    xc = x[:, ~frozen] if (frozen.any() and (~frozen).sum() > 2) else x
    cmin, cmax = xc.min(axis=1), xc.max(axis=1)
    n_hit = (xc == cmax[:, None]).sum(axis=1) + (xc == cmin[:, None]).sum(axis=1)
    clip_frac = np.where(cmax > cmin, n_hit / xc.shape[1], 0.0)

    corr = spatial_correlation(rec, cfg) if corr is None else corr
    w = max(1, int(min(60.0, n / rec.sfreq / 2.0) * rec.sfreq))
    df = pd.DataFrame({
        "channel": rec.ch_names,
        "std_uv": xd.std(axis=1),
        "std_clean_uv": xc.std(axis=1),
        "window_std_uv": window_std_uv(xd, rec.sfreq),
        "variance_uv2": xd.var(axis=1),
        "skewness": stats.skew(xd, axis=1),
        "kurtosis": stats.kurtosis(xd, axis=1),
        "median_uv": med,
        "p_low_uv": p_lo, "p_high_uv": p_hi, "range_p_uv": p_hi - p_lo,
        "ptp_uv": xd.max(axis=1) - xd.min(axis=1),
        "clip_frac": clip_frac,
        "longest_equal_run_s": [longest_equal_run(xc[i]) / rec.sfreq for i in range(n_ch)],
        "drift_uv": np.median(x[:, -w:], axis=1) - np.median(x[:, :w], axis=1),
        "corr_median": corr["median"],
        "corr_low_frac": corr["low_frac"],
    })
    med_win = float(df["window_std_uv"].median())
    df["flat"] = df["std_clean_uv"] < cfg.flat_std_uv
    df["clipped"] = (~df["flat"]) & (df["clip_frac"] > cfg.clip_fraction)
    df["noisy"] = ((~df["flat"]) & (~df["clipped"])
                   & (df["window_std_uv"] > np.maximum(cfg.noisy_ratio * med_win, cfg.noisy_min_uv)))
    df["corr_threshold_used"] = corr["thresholds"]
    df["corr_below_thr"] = (~df["flat"]) & (~df["clipped"]) & (df["corr_median"] < df["corr_threshold_used"])
    unreliable = bool(df["corr_below_thr"].mean() > cfg.corr_unreliable_fraction)
    df["corr_ref_unreliable"] = unreliable
    df["uncorrelated"] = df["corr_below_thr"] & (not unreliable)
    df["bad"] = df["flat"] | df["clipped"] | df["noisy"] | df["uncorrelated"]
    crit = {_norm_name(k) for k in cfg.critical_channels}
    df["is_critical"] = [_norm_name(nm) in crit for nm in rec.ch_names]
    df["dc_offset_info"] = df["median_uv"].abs() > cfg.dc_offset_info_uv
    df["drift_info"] = df["drift_uv"].abs() > cfg.drift_info_uv
    return df


def integrity_check(rec: Recording, cfg: QCConfig, ch_table: pd.DataFrame) -> dict[str, Any]:
    """Layer 2: flat, clipped, noisy and uncorrelated channels, frozen segments, and what can be repaired.

    * 2 or more CRITICAL channels bad                               -> FAIL (CRITICAL_CHANNELS_BAD)
    * more bad channels than ``cfg.max_bad_for(n_eeg)``              -> FAIL (TOO_MANY_BAD_CHANNELS)
    * 1 critical channel bad                                         -> FLAG (CRITICAL_CHANNEL_BAD)
    * 1..max bad channels, nothing worse                             -> PASS, label USABLE_NEEDS_INTERPOLATION
    """
    t = ch_table
    n_eeg = len(t)
    max_bad = cfg.max_bad_for(n_eeg)
    flat, clipped, noisy, uncorr = (t.loc[t[k], "channel"].tolist() for k in ("flat", "clipped", "noisy", "uncorrelated"))
    bad_rows = t[t["bad"]]
    bad = bad_rows["channel"].tolist()
    n_bad = len(bad)
    crit_bad = bad_rows.loc[bad_rows["is_critical"], "channel"].tolist()
    other_bad = [b for b in bad if b not in crit_bad]
    detail = ", ".join(
        f"{r.channel} ({'+'.join(k for k in ('flat', 'clipped', 'noisy', 'uncorrelated') if getattr(r, k))})"
        for r in bad_rows.itertuples())
    crit_present = [c for c in cfg.critical_channels if find_channel(rec.ch_names, c) is not None]

    issues: list[dict[str, Any]] = []
    crit_txt = ", ".join(cfg.critical_channels)
    if len(crit_bad) >= cfg.critical_bad_fail_min:
        issues.append(make_issue("FAIL", "critical_channels_bad", ", ".join(crit_bad),
                                 f"fewer than {cfg.critical_bad_fail_min} of {crit_txt} bad", code="CRITICAL_CHANNELS_BAD"))
    elif len(crit_bad) == 1:
        issues.append(make_issue("FLAG", "critical_channel_bad", crit_bad[0],
                                 f"none of {crit_txt} bad (one can be interpolated)", code="CRITICAL_CHANNEL_BAD"))
    if n_bad > max_bad:
        issues.append(make_issue("FAIL", "bad_channels_total", f"{n_bad}: {detail}",
                                 f"at most {max_bad} of {n_eeg} channels (more cannot be interpolated reliably)",
                                 code="TOO_MANY_BAD_CHANNELS"))
    interpolation_needed = n_bad > 0 and not any(i["verdict"] == "FAIL" for i in issues)
    if interpolation_needed:
        issues.append(make_issue("PASS", "bad_channels_to_interpolate", detail,
                                 f"1-{max_bad} bad channels can be interpolated; the recording stays usable",
                                 code="USABLE_NEEDS_INTERPOLATION"))
    if bool(t["corr_ref_unreliable"].any()):
        share = float(t["corr_below_thr"].mean())
        issues.append(make_issue(
            "FLAG", "spatial_correlation_reference_unreliable",
            f"{share:.0%} of channels below their r threshold ({cfg.corr_threshold} x {cfg.corr_scale(n_eeg):.2f}; "
            f"{cfg.corr_threshold_relaxed} x {cfg.corr_scale(n_eeg):.2f} for O1/O2/Oz)",
            f"at most {cfg.corr_unreliable_fraction:.0%} of channels below the threshold "
            "(otherwise the correlation criterion is not applied)", code="LOW_GLOBAL_SPATIAL_CORRELATION"))

    _, segs = find_frozen(rec.data_uv, rec.sfreq, cfg)
    frozen_s = float(sum(length for _, length in segs))
    if frozen_s > cfg.global_frozen_fail_fraction * rec.duration_s:
        issues.append(make_issue("FAIL", "all_channels_frozen_s", frozen_s,
                                 f"less than {cfg.global_frozen_fail_fraction:.0%} of the recording", code="RECORDING_FROZEN"))
    elif frozen_s > cfg.global_frozen_flag_s:
        issues.append(make_issue("FLAG", "all_channels_frozen_s", frozen_s, f"at most {cfg.global_frozen_flag_s} s",
                                 code="FROZEN_SEGMENT"))

    metrics = {
        "n_eeg_channels": n_eeg, "max_bad_allowed": max_bad, "corr_scale": cfg.corr_scale(n_eeg),
        "n_bad_channels": n_bad, "bad_channels": bad, "bad_detail": detail,
        "n_flat": len(flat), "n_clipped": len(clipped), "n_noisy": len(noisy), "n_uncorrelated": len(uncorr),
        "flat_channels": flat, "clipped_channels": clipped, "noisy_channels": noisy, "uncorrelated_channels": uncorr,
        "n_critical_bad": len(crit_bad), "critical_bad_channels": crit_bad, "n_noncritical_bad": len(other_bad),
        "critical_channels_present": crit_present,
        "interpolation_needed": bool(interpolation_needed),
        "median_spatial_corr": float(t["corr_median"].median()),
        "corr_reference_reliable": not bool(t["corr_ref_unreliable"].any()),
        "max_clip_fraction": float(t["clip_frac"].max()),
        "frozen_s": round(frozen_s, 3),
        "frozen_segments": "; ".join(f"{a:.1f}s+{b:.1f}s" for a, b in segs[:5]),
    }
    return make_result("integrity", metrics, issues)


# ================================================================================ LAYER 3: amplitude
def amplitude_check(cfg: QCConfig, ch_table: pd.DataFrame) -> dict[str, Any]:
    """Layer 3: is the amplitude in a physiological range? Slow drift cannot inflate it (baseline-corrected)."""
    t = ch_table
    typical = float(t["window_std_uv"].median())
    frac_high = float((t["window_std_uv"] > cfg.typical_amp_max_uv).mean())
    issues = []
    if not (cfg.typical_amp_min_uv <= typical <= cfg.typical_amp_max_uv):
        issues.append(make_issue("FAIL", "typical_amplitude_uv", typical,
                                 f"between {cfg.typical_amp_min_uv:.0f} and {cfg.typical_amp_max_uv:.0f} uV (probable unit/scale problem)",
                                 code="AMPLITUDE_OUT_OF_RANGE"))
    elif frac_high > cfg.high_amp_channel_fraction:
        issues.append(make_issue("FLAG", "fraction_channels_high_amplitude", frac_high,
                                 f"at most {cfg.high_amp_channel_fraction:.0%} of channels above {cfg.typical_amp_max_uv:.0f} uV",
                                 code="HIGH_AMPLITUDE_CHANNELS"))
    metrics = {
        "typical_amplitude_uv": typical,
        "whole_recording_std_uv": float(t["std_uv"].median()),
        "median_variance_uv2": float(t["variance_uv2"].median()),
        "median_skewness": float(t["skewness"].median()),
        "median_kurtosis": float(t["kurtosis"].median()),
        "median_range_0.5_99.5_uv": float(t["range_p_uv"].median()),
        "fraction_high_amplitude": frac_high,
        "n_dc_offset_info": int(t["dc_offset_info"].sum()),
        "n_drift_info": int(t["drift_info"].sum()),
    }
    return make_result("amplitude", metrics, issues)


# ================================================================================ LAYER 4: spectral
def compute_psd(rec: Recording, cfg: QCConfig) -> dict[str, Any]:
    """Welch PSD (uV^2/Hz) of ALL EEG channels, in memory only. Hann window, 4 s, 50 % overlap."""
    nperseg = min(int(round(cfg.welch_window_s * rec.sfreq)), rec.n_times)
    freqs, psd = signal.welch(rec.data_uv, fs=rec.sfreq, window="hann", nperseg=nperseg,
                              noverlap=int(nperseg * cfg.welch_overlap), detrend="constant", axis=-1)
    return {"freqs": freqs, "psd": psd.astype(np.float32), "channels": list(rec.ch_names), "sfreq": rec.sfreq}


def _band(f: np.ndarray, band) -> np.ndarray:
    return (f >= band[0]) & (f <= band[1])


def _flank_mean(f: np.ndarray, p: np.ndarray, flanks) -> float:
    m = np.zeros(f.shape, dtype=bool)
    for b in flanks:
        m |= _band(f, b)
    return float(p[m].mean()) if m.any() else float("nan")


def alpha_metrics(f: np.ndarray, p: np.ndarray, cfg: QCConfig) -> dict[str, float]:
    """Alpha peak frequency, prominence (dB over the flanks) and band power (uV^2) of one PSD."""
    m = _band(f, cfg.alpha_band_hz)
    if not m.any():
        return {"peak_hz": np.nan, "prominence_db": np.nan, "power_uv2": np.nan}
    i = int(np.argmax(p[m]))
    peak, flank = float(p[m][i]), _flank_mean(f, p, cfg.alpha_flank_hz)
    prom = 10 * np.log10(peak / flank) if (flank > 0 and peak > 0) else np.nan
    return {"peak_hz": float(f[m][i]), "prominence_db": float(prom), "power_uv2": float(trapezoid(p[m], f[m]))}


def line_db(f: np.ndarray, p: np.ndarray, center: float, cfg: QCConfig, nyq: float) -> float:
    """Line noise in dB: highest PSD near ``center`` versus the mean PSD of the two flanking bands.

    A flank that would reach beyond 98 % of the Nyquist frequency is dropped (or clipped); if nothing is left
    the line is not assessable and NaN is returned.
    """
    lo, hi = cfg.line_flank_offsets_hz
    hw, edge = cfg.line_peak_halfwidth_hz, 0.98 * nyq
    if center is None or center + hw > edge:
        return float("nan")
    m = _band(f, (center - hw, center + hw))
    flanks = [(center - hi, center - lo)]
    if center + hi <= edge:
        flanks.append((center + lo, center + hi))
    elif edge > center + lo:
        flanks.append((center + lo, edge))
    flank = _flank_mean(f, p, flanks)
    if not m.any() or not (flank > 0):
        return float("nan")
    return float(10 * np.log10(p[m].max() / flank))


def resolve_line_freq(psd: dict[str, Any], cfg: QCConfig, sidecar: dict | None) -> dict[str, Any]:
    """Mains frequency of this recording. Priority: explicit config/CLI, JSON sidecar ``PowerLineFrequency``,
    then the dominant of the candidate frequencies (50 / 60 Hz) in the spectrum (median over channels)."""
    nyq = psd["sfreq"] / 2.0
    if cfg.forced_line_hz is not None:
        return {"hz": cfg.forced_line_hz, "source": "config", "candidates_db": {}}
    pl = (sidecar or {}).get("PowerLineFrequency")
    if isinstance(pl, (int, float)) and not isinstance(pl, bool):
        for cand in cfg.line_candidates_hz:
            if abs(float(pl) - cand) < 0.5:
                return {"hz": float(cand), "source": "sidecar", "candidates_db": {}}
    f, cands = psd["freqs"], {}
    for cand in cfg.line_candidates_hz:
        vals = [line_db(f, p.astype(float), cand, cfg, nyq) for p in psd["psd"]]
        vals = [v for v in vals if np.isfinite(v)]
        cands[float(cand)] = float(np.median(vals)) if vals else float("nan")
    finite = {k: v for k, v in cands.items() if np.isfinite(v)}
    if not finite:
        return {"hz": None, "source": "not_assessable", "candidates_db": cands}
    best = max(finite, key=finite.get)
    if finite[best] < cfg.line_detect_min_db:
        # no mains line stands out: keep the first assessable candidate so the (low) numbers are still reported
        return {"hz": next(iter(finite)), "source": "default_no_line_detected", "candidates_db": cands}
    return {"hz": best, "source": "auto", "candidates_db": cands}


def channel_spectral_table(psd: dict[str, Any], cfg: QCConfig, line_hz: float | None) -> pd.DataFrame:
    """Alpha power/prominence and line noise for EVERY channel (used for the figures)."""
    f, rows, nyq = psd["freqs"], [], psd["sfreq"] / 2.0
    for p in psd["psd"]:
        p = p.astype(float)
        a = alpha_metrics(f, p, cfg)
        rows.append({"alpha_power_uv2": a["power_uv2"], "alpha_prominence_db": a["prominence_db"],
                     "line_db": line_db(f, p, line_hz, cfg, nyq) if line_hz else float("nan")})
    return pd.DataFrame(rows)


def occipital_metrics(psd: dict[str, Any], cfg: QCConfig, line_hz: float | None) -> dict[str, Any]:
    """Alpha, line-noise and broadband numbers on the posterior ROI (O1, O2, Oz; O1/O2 on caps without Oz)."""
    nyq = psd["sfreq"] / 2.0
    idx = {t: find_channel(psd["channels"], t) for t in cfg.psd_channels}
    idx = {t: i for t, i in idx.items() if i is not None}
    f, out = psd["freqs"], {"channels": list(idx)}
    lo, hi = cfg.dead_spectrum_band_hz
    band = _band(f, (lo, cfg.band_hi(hi, psd["sfreq"])))
    peaks, proms, powers, line0, harm, broad = [], [], [], [], {o: [] for o in cfg.harmonic_orders}, []
    for t, i in idx.items():
        p = psd["psd"][i].astype(float)
        a = alpha_metrics(f, p, cfg)
        out[f"alpha_power_uv2_{t}"] = a["power_uv2"]
        peaks.append(a["peak_hz"])
        proms.append(a["prominence_db"])
        powers.append(a["power_uv2"])
        line0.append(line_db(f, p, line_hz, cfg, nyq) if line_hz else np.nan)
        for o in cfg.harmonic_orders:
            harm[o].append(line_db(f, p, line_hz * o, cfg, nyq) if line_hz else np.nan)
        broad.append(float(trapezoid(p[band], f[band])) if band.any() else float("nan"))

    def agg(v, fn):
        v = [x for x in v if np.isfinite(x)]
        return float(fn(v)) if v else float("nan")

    out.update(alpha_peak_hz=agg(peaks, np.mean), alpha_prominence_db=agg(proms, np.median),
               alpha_power_uv2=agg(powers, np.mean), line_hz=line_hz if line_hz else float("nan"),
               line_db=agg(line0, np.max), broadband_power_uv2=agg(broad, np.median),
               dead_spectrum=bool(len(broad) > 0 and all(np.nan_to_num(b, nan=0.0) < cfg.dead_spectrum_power_uv2 for b in broad)))
    for o in cfg.harmonic_orders:
        out[f"line_h{o}_db"] = agg(harm[o], np.max)
    return out


def alpha_ratio_info(o_closed: dict[str, Any] | None, o_open: dict[str, Any] | None) -> dict[str, Any]:
    """Alpha power Eyes Closed / Eyes Open per posterior channel and its median (one EC/EO pair)."""
    ratios: dict[str, float] = {}
    if o_closed and o_open:
        for t in o_closed["channels"]:
            if t in o_open["channels"]:
                a_eo = nan_if_none(o_open.get(f"alpha_power_uv2_{t}"))
                a_ec = nan_if_none(o_closed.get(f"alpha_power_uv2_{t}"))
                ratios[t] = float(a_ec / a_eo) if a_eo > 0 else float("nan")
    valid = [v for v in ratios.values() if not np.isnan(v)]
    return {"applicable": True, "valid": valid, "median": float(np.median(valid)) if valid else float("nan")}


def _occ_from_json(o: dict[str, Any] | None) -> dict[str, Any] | None:
    """Posterior-ROI metrics as stored in JSON (NaN is ``None`` there) back to floats."""
    if not o:
        return None
    return {k: (v if k in ("channels", "dead_spectrum") else nan_if_none(v)) for k, v in o.items()}


def spectral_layer(o: dict[str, Any] | None, condition: str, ratio: dict[str, Any], clipped_frac: float,
                   cfg: QCConfig) -> dict[str, Any]:
    """Layer 4 for ONE recording, given the posterior-ROI metrics ``o`` and the EC/EO ratio of its pair.

    * Dead spectrum (no power at all on the posterior ROI): the only alpha-related FAIL.
    * Alpha blocking and alpha prominence are FLAGS only: 10-15 % of healthy adults are alpha non-responders.
    * Line noise above ``line_flag_db`` / ``line_high_db`` is a FLAG (a notch filter fixes it); a FAIL only
      together with clipping of at least half of the channels (LINE_NOISE_SATURATION).
    """
    o = _occ_from_json(o)
    issues: list[dict[str, Any]] = []
    if not o or not o["channels"]:
        issues.append(make_issue("FAIL", "occipital_channels_missing", "none of the posterior ROI channels found",
                                 f"{', '.join(cfg.psd_channels)} needed", code="OCCIPITAL_CHANNELS_MISSING"))
        return make_result("spectral", {"n_psd_channels": 0}, issues)
    metrics = {k: v for k, v in o.items() if k != "channels" and not k.startswith("alpha_power_uv2_")}
    metrics["posterior_roi"] = o["channels"]
    metrics["condition"] = condition
    line_unit = int(round(o["line_hz"])) if np.isfinite(o["line_hz"]) else None
    metrics["notch_suggested"] = bool(np.isfinite(o["line_db"]) and o["line_db"] > cfg.notch_suggested_db)

    if o["dead_spectrum"]:
        lo, hi = cfg.dead_spectrum_band_hz
        issues.append(make_issue("FAIL", "occipital_broadband_power_uv2", o["broadband_power_uv2"],
                                 f"at least {cfg.dead_spectrum_power_uv2} uV^2 in {lo:.0f}-{hi:.0f} Hz on the posterior ROI (spectrum is flat / dead)",
                                 code="DEAD_SPECTRUM"))
    db, sat = o["line_db"], float(clipped_frac)
    if np.isfinite(db):
        if db > cfg.line_high_db:
            if sat >= cfg.line_saturation_channel_fraction:
                issues.append(make_issue("FAIL", f"line_{line_unit}Hz_with_saturation", f"{db:.1f} dB, {sat:.0%} of channels clipped",
                                         f"at most {cfg.line_high_db:.0f} dB, or fewer than {cfg.line_saturation_channel_fraction:.0%} of channels clipped",
                                         code="LINE_NOISE_SATURATION"))
            else:
                issues.append(make_issue("FLAG", f"line_{line_unit}Hz_db", db, f"at most {cfg.line_high_db:.0f} dB (notch filter needed)",
                                         code="HIGH_LINE_NOISE"))
        elif db > cfg.line_flag_db:
            issues.append(make_issue("FLAG", f"line_{line_unit}Hz_db", db, f"at most {cfg.line_flag_db:.0f} dB", code="LINE_NOISE"))

    weak_peak = (condition == "closed" and np.isfinite(o["alpha_prominence_db"])
                 and o["alpha_prominence_db"] < cfg.alpha_prominence_min_db)
    if weak_peak:
        issues.append(make_issue("FLAG", "alpha_peak_prominence_db", o["alpha_prominence_db"],
                                 f"at least {cfg.alpha_prominence_min_db:.0f} dB in Eyes Closed", code="LOW_ALPHA_PROMINENCE"))
    valid = ratio.get("valid", []) if ratio.get("applicable") else []
    ratio_med = ratio.get("median", float("nan")) if ratio.get("applicable") else float("nan")
    metrics["alpha_ratio_EC_EO"] = ratio_med
    if ratio.get("applicable") and condition in ("closed", "open"):
        if valid:
            if ratio_med < cfg.alpha_ratio_pass:
                if max(valid) < cfg.alpha_no_blocking_ratio:
                    issues.append(make_issue("FLAG", "alpha_ratio_EC_EO", ratio_med,
                                             f"at least {cfg.alpha_no_blocking_ratio} on one channel", code="NO_ALPHA_BLOCKING"))
                else:
                    issues.append(make_issue("FLAG", "alpha_ratio_EC_EO", ratio_med,
                                             f"median at least {cfg.alpha_ratio_pass}", code="WEAK_ALPHA_BLOCKING"))
        else:
            issues.append(make_issue("FLAG", "alpha_ratio_EC_EO", "not computed",
                                     "needs both Eyes Closed and Eyes Open", code="ALPHA_RATIO_NOT_COMPUTED"))
    metrics["alpha_non_responder_possible"] = bool(condition == "closed" and weak_peak and valid and ratio_med < cfg.alpha_ratio_pass)
    return make_result("spectral", metrics, issues)


# ================================================================================ LAYER 5: consistency
def consistency_layer(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """All recordings of a subject should have the same channel names (and order) and sampling rate.

    ``summaries`` maps rec_id -> {"ch_names": [...], "sfreq": float}. A recording that differs from the most
    common layout gets a FAIL.
    """
    if not summaries:
        return {}
    ref_names = Counter(tuple(s["ch_names"]) for s in summaries.values()).most_common(1)[0][0]
    ref_sf = Counter(s["sfreq"] for s in summaries.values()).most_common(1)[0][0]
    out = {}
    for key, s in summaries.items():
        issues = []
        if tuple(s["ch_names"]) != ref_names:
            issues.append(make_issue("FAIL", "channel_names_or_order_differ", "different", "same as the other recordings",
                                     code="CHANNEL_LAYOUT_DIFFERS"))
        if s["sfreq"] != ref_sf:
            issues.append(make_issue("FAIL", "sfreq_differs", s["sfreq"], f"same as the other recordings ({ref_sf})",
                                     code="SFREQ_DIFFERS"))
        out[key] = make_result("consistency", {"same_layout_as_cohort": not issues}, issues)
    return out


# ================================================================================ events (information only)
def stim_events(rec: Recording) -> dict[str, Any]:
    """Summarise trigger channels: number of events, codes, first/last time. Codes are NOT interpreted."""
    out: dict[str, Any] = {"stim_channels": ", ".join((rec.stim_data or {}).keys()), "stim_n_events": 0,
                           "stim_codes": "", "stim_first_event_s": np.nan, "stim_last_event_s": np.nan}
    if not rec.stim_data:
        return out
    codes, first, last, n_ev = [], [], [], 0
    for v in rec.stim_data.values():
        vals = np.rint(v).astype(np.int64)
        uniq, cnt = np.unique(vals, return_counts=True)
        baseline = int(uniq[np.argmax(cnt)])
        change = np.flatnonzero(np.diff(vals) != 0) + 1
        ev = change[vals[change] != baseline]
        n_ev += len(ev)
        codes += [int(c_) for c_ in vals[ev]]
        if len(ev):
            first.append(ev.min() / rec.sfreq)
            last.append(ev.max() / rec.sfreq)
    out.update(stim_n_events=n_ev, stim_codes="; ".join(f"{k} x{v}" for k, v in Counter(codes).most_common(10)),
               stim_first_event_s=round(min(first), 2) if first else np.nan,
               stim_last_event_s=round(max(last), 2) if last else np.nan)
    return out


def events_info(rec: Recording, events_path: Path | None) -> dict[str, Any]:
    info = {"annotations": len(rec.annotations), **stim_events(rec), "events_tsv_rows": np.nan}
    if events_path is not None and Path(events_path).exists():
        try:
            info["events_tsv_rows"] = len(pd.read_csv(events_path, sep="\t"))
        except Exception as e:
            info["events_tsv_rows"] = f"unreadable ({type(e).__name__})"
    return info


# ================================================================================ one recording
def analyse_recording(rec: Recording, cfg: QCConfig, sidecar: dict | None = None, sidecar_error: str | None = None,
                      events_path: Path | None = None, file_kb: float | None = None) -> dict[str, Any]:
    """Run layers 1-3 and compute the posterior-ROI spectrum of ONE recording.

    The pair-dependent part of layer 4 and layer 5 are completed later by :func:`finalize_records`.
    Returns a JSON-friendly dict (plus ``ch_table`` as a DataFrame and ``psd_occ`` arrays for the figures).
    """
    montage = check_montage(rec.raw_ch_names, cfg)
    header = header_check(rec, cfg, montage, sidecar, sidecar_error, file_kb)
    base = {"n_eeg": rec.n_eeg, "montage": {"supported": montage.supported, "fraction": montage.fraction,
                                            "anchors_found": montage.anchors_found},
            "summary": {"ch_names": list(rec.ch_names), "sfreq": rec.sfreq}, "info": events_info(rec, events_path)}
    if not montage.supported:
        return {**base, "status": "UNSUPPORTED", "layers": {"header": header}, "ch_table": None, "occ": None,
                "psd_occ": None, "line": {"hz": None, "source": "not_assessed"}, "clipped_frac": 0.0}

    try:
        corr = spatial_correlation(rec, cfg)
        ch = compute_channel_table(rec, cfg, corr)
    finally:
        rec.release_dt()
        gc.collect()
    psd = compute_psd(rec, cfg)
    line = resolve_line_freq(psd, cfg, sidecar)
    ch = pd.concat([ch, channel_spectral_table(psd, cfg, line["hz"])], axis=1)
    layers = {"header": header, "integrity": integrity_check(rec, cfg, ch), "amplitude": amplitude_check(cfg, ch)}
    occ = occipital_metrics(psd, cfg, line["hz"])
    keep = _band(psd["freqs"], (0.0, 100.0))
    roi = [find_channel(psd["channels"], t) for t in cfg.psd_channels]
    roi = [i for i in roi if i is not None]
    psd_occ = {"freqs": psd["freqs"][keep].astype(np.float32),
               "psd": psd["psd"][roi][:, keep] if roi else np.zeros((0, int(keep.sum())), dtype=np.float32),
               "channels": [psd["channels"][i] for i in roi]}
    return {**base, "status": "OK", "layers": layers, "ch_table": ch, "occ": occ, "psd_occ": psd_occ, "line": line,
            "clipped_frac": float(ch["clipped"].mean())}


# ================================================================================ subject level
def recording_label(overall: str, layers: dict[str, Any]) -> str:
    """Screening label of one recording. Only a FAIL is rejected; repairable bad channels are not."""
    if overall == "MISSING":
        return "MISSING"
    if overall == "FAIL":
        return "REJECT"
    if layers.get("integrity", {}).get("metrics", {}).get("interpolation_needed", False):
        return "USABLE_NEEDS_INTERPOLATION"
    return "USABLE_WITH_FLAGS" if overall == "FLAG" else "USABLE"


def recording_codes(rec: dict[str, Any]) -> list[str]:
    if rec["status"] == "ERROR":
        return ["FILE_NOT_READABLE"]
    if rec["status"] == "MISSING":
        return [rec.get("error_code") or "RECORDING_MISSING"]
    return sorted({i["code"] for lay in rec["layers"].values() for i in lay["issues"]})


def finalize_records(records: list[dict[str, Any]], cfg: QCConfig) -> list[dict[str, Any]]:
    """Complete layers 4 (EC/EO pairing) and 5 (consistency) for all recordings of ONE subject and set the
    overall verdict, label and codes. Works on copies; the cached per-recording results are not modified."""
    recs = [copy.deepcopy(r) for r in records]
    ok = [r for r in recs if r["status"] == "OK"]
    conds = {r["condition"] for r in recs if r["status"] in ("OK", "UNSUPPORTED", "ERROR", "MISSING")}
    paired_design = {"closed", "open"} <= conds

    groups: dict[tuple, dict[str, dict]] = {}
    for r in ok:
        groups.setdefault(tuple(r["pair_key"]), {})[r["condition"]] = r
    for r in ok:
        cond = r["condition"]
        if cond in ("closed", "open"):
            partner = groups[tuple(r["pair_key"])].get("open" if cond == "closed" else "closed")
            if partner is not None:
                oc, oo = (r["occ"], partner["occ"]) if cond == "closed" else (partner["occ"], r["occ"])
                ratio = alpha_ratio_info(oc, oo)
            else:
                ratio = {"applicable": paired_design, "valid": [], "median": float("nan")}
        else:
            ratio = {"applicable": False, "valid": [], "median": float("nan")}
        r["layers"]["spectral"] = spectral_layer(r["occ"], cond, ratio, r.get("clipped_frac", 0.0), cfg)

    cons = consistency_layer({r["rec_id"]: r["summary"] for r in ok})
    for r in ok:
        r["layers"]["consistency"] = cons[r["rec_id"]]

    for r in recs:
        layers = r["layers"]
        verdicts = {f"verdict_{k}": (layers[k]["verdict"] if k in layers else "N/A") for k in LAYER_ORDER}
        if r["status"] == "MISSING":
            overall = "MISSING"
        elif r["status"] == "ERROR" or r.get("error"):
            overall = "FAIL"
        else:
            overall = worst_verdict(verdicts.values())
        r["verdicts"] = verdicts
        r["overall"] = overall
        r["label"] = recording_label(overall, layers)
        r["codes"] = recording_codes(r)
    return recs


def subject_label(labels: list[str]) -> str:
    labels = [lab for lab in labels if lab in LABEL_RANK]
    if not labels:
        return "NO DATA"
    if all(lab == "REJECT" for lab in labels):
        return "REJECT"
    if any(lab == "REJECT" for lab in labels):
        return "PARTIALLY_USABLE"
    return max(labels, key=LABEL_RANK.__getitem__)


def metric(layers: dict, layer: str, name: str, default: Any = np.nan) -> Any:
    return layers.get(layer, {}).get("metrics", {}).get(name, default)
