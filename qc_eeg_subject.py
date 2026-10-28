#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_eeg_subject.py - first-pass SCREENING (triage) of the resting-state EEG of ONE OR MANY subjects
OpenNeuro ds005385 (Dortmund Vital Study), dataset version 1.0.3
==========================================================================

Goal: keep every healthy, usable recording and throw away only the catastrophic, unsalvageable ones.

Which subjects?  Type numbers only (no "sub-" needed):
    python qc_eeg_subject.py                  # asks which subjects
    python qc_eeg_subject.py 18               # one subject
    python qc_eeg_subject.py 1 2 3            # several subjects, separated by spaces
    python qc_eeg_subject.py 1 to 20          # a range
    python qc_eeg_subject.py 1 5 to 8 12      # mixed: 1, 5, 6, 7, 8, 12
The same text can be typed when the script asks "Subjects >".

What happens for every subject
------------------------------
1. Eligibility (embedded participant table; nothing is downloaded yet):
       * age > 60                         -> REJECTED
       * late triggers > 0 (any session)  -> REJECTED
       * only session 1 completed         -> fine, only ses-1 (4 recordings) is analysed
   If the subject was already analysed with the same QC logic and thresholds, the earlier results are reused:
   nothing is downloaded and nothing is analysed again (--force-reanalysis overrides this).
2. Download of exactly the needed files (DataLad like the manual workflow; plain HTTPS as fallback).
   Files that are already on disk are not downloaded again.
3. QC layers on every recording that exists (a missing recording is reported as MISSING):
       header / metadata - channel integrity (flat, clipped, noisy, spatially uncorrelated, frozen) -
       amplitude - spectral validation (alpha, 50 Hz, dead spectrum) - consistency.
   Voltage statistics use a baseline-corrected copy of the signal (0.5 Hz high-pass, in memory only).
4. Tables, figures (A-G, channel quality topomap), report and run manifest.
5. Clean-up: the analysed data are deleted (data folder + DataLad copy) to save disk space (--keep-data keeps them).

Screening rules (all thresholds are in CONFIG below)
----------------------------------------------------
    FAIL only for catastrophic problems:
        * 2 or more critical channels bad (Cz, Fz, Pz, O1, O2; e.g. both O1 and O2)
        * more than 6 bad channels in total (too many to interpolate)
        * dead spectrum on O1/O2/Oz, recording shorter than 1 min, wrong scale, frozen recording,
          unreadable file, metadata errors, 50 Hz noise together with clipping of half of the channels
    FLAG (kept, but worth a look): weak alpha / no alpha blocking (alpha non-responders are healthy),
        50 Hz line noise (HIGH_LINE_NOISE above 30 dB: a notch filter removes it), one bad critical channel, ...
    PASS with label USABLE_NEEDS_INTERPOLATION: 1-6 bad channels, at most one of them critical.
    Per recording: label USABLE | USABLE_WITH_FLAGS | USABLE_NEEDS_INTERPOLATION | REJECT, plus reason codes.
    Per subject: the most demanding label, or PARTIALLY_USABLE when some recordings are usable and others rejected.

Where the results go (default --out-root qc_output)
---------------------------------------------------
    qc_output/
        summary_all_subjects.csv      one line per subject processed so far (updated by every run)
        sub-018/
            status.txt                the whole story of this subject in plain text (also for rejected ones)
            status.json               the same, machine-readable
            qc_results/               tables (csv), figures/, qc_report.txt, run_manifest.json, run_log.txt
        sub-002/
            status.txt                e.g. "REJECTED: Age above 60 (67)"   (no qc_results folder, nothing downloaded)
        ...
    The channel quality topomap is  qc_results/figures/sub-<ID>_task-rest_channel_quality_topomap.png

A problem in one subject never stops the others: it is written into that subject's status file.
Press Ctrl+C to stop; everything that was finished stays saved.

Other options
-------------
    --data-dir data  --repo-dir ds005385_repo  --out-root qc_output
    --no-download            use the EDF files that are already in the data folder (they are never deleted)
    --keep-data              do not delete the downloaded data after the analysis
    --force-reanalysis       analyse again even if results of the same QC version exist
    --force-download         download again even if the files are on disk
    --verify-table FILE.tsv  compare the embedded participant table with your sheet (TSV) and exit
    --debug                  show the full traceback and stop at the first problem

Requirements
------------
    pip install "mne>=1.6" edfio numpy scipy pandas matplotlib
    DataLad + git-annex (recommended; same as the manual workflow)

Exit codes: 0 = everything finished, 1 = a technical problem (error, no data, Ctrl+C),
            2 = invalid input, or every requested subject was rejected.

Nothing is ever written into the raw data. Signals are only filtered/detrended in memory
for statistics. All thresholds are in CONFIG below.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")  # no window: figures are only saved to disk
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LogNorm, Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)
    from scipy import signal, stats
    from scipy.integrate import trapezoid
except ImportError as _e:  # pragma: no cover
    sys.exit(f"Missing Python package: {_e.name}\n"
             'Install the requirements with:\n  pip install "mne>=1.6" edfio numpy scipy pandas matplotlib')

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================================================================
# TERMINAL OUTPUT: colours, symbols and a small "working..." spinner.
# The log file receives the same text without colours.
# ======================================================================
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_REAL_STDOUT = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
_IS_TTY = bool(getattr(_REAL_STDOUT, "isatty", lambda: False)())
USE_COLOR = _IS_TTY and not os.environ.get("NO_COLOR") and (os.name != "nt" or "WT_SESSION" in os.environ)
_UTF8 = (getattr(_REAL_STDOUT, "encoding", "") or "").lower().replace("-", "").startswith("utf")
SYM = ({"ok": "✔", "fail": "✘", "warn": "!", "arrow": "→", "spin": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"} if _UTF8
       else {"ok": "OK", "fail": "X", "warn": "!", "arrow": "->", "spin": "|/-\\"})
_STYLE = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36",
          "bold": "1", "dim": "2"}
STATUS_COLOR = {"PASS": "green", "FLAG": "yellow", "FAIL": "red", "MISSING": "magenta", "NO DATA": "red",
                "COMPLETED": "green", "REJECTED": "yellow", "NO_DATA": "red", "ERROR": "red",
                "INTERRUPTED": "yellow", "INFO": "cyan",
                "USABLE": "green", "USABLE_WITH_FLAGS": "yellow", "USABLE_NEEDS_INTERPOLATION": "cyan", "REJECT": "red",
                "PARTIALLY_USABLE": "yellow"}


def paint(text: str, *styles: str) -> str:
    if not USE_COLOR or not styles:
        return text
    return f"\x1b[{';'.join(_STYLE[s] for s in styles)}m{text}\x1b[0m"


def vc(value: Any) -> str:
    """A verdict / status word in its colour (PASS green, FLAG yellow, FAIL red ...)."""
    s = str(value)
    return paint(s, STATUS_COLOR.get(s, "dim"), "bold")


def term_width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 20)).columns, 130))


def say(text: str = "", indent: int = 0) -> None:
    print(" " * indent + text)


def good(text: str, indent: int = 4) -> None:
    say(f"{paint(SYM['ok'], 'green', 'bold')} {text}", indent)


def warn(text: str, indent: int = 4) -> None:
    say(f"{paint(SYM['warn'], 'yellow', 'bold')} {text}", indent)


def bad(text: str, indent: int = 4) -> None:
    say(f"{paint(SYM['fail'], 'red', 'bold')} {text}", indent)


def note(text: str, indent: int = 4) -> None:
    say(paint(f"{SYM['arrow']} {text}", "dim"), indent)


def step(n: int, total: int, title: str) -> None:
    print()
    say(f"{paint(f'Step {n}/{total}', 'cyan', 'bold')}  {paint(title, 'bold')}", 2)


def rule(char: str = "=") -> None:
    say(paint(char * term_width(), "dim"))


class Task:
    """Shows a spinner with the elapsed time while something runs, then one result line.

        with Task("Downloading file X") as t:
            ...
            t.detail = "17.9 MB"        # shown after the label
            # t.fail("reason")           # marks the line as failed (cross instead of tick)

    When the output is not a terminal (pipe, log file) the label is printed once instead of a spinner.
    """

    def __init__(self, label: str, indent: int = 4) -> None:
        self.label, self.indent = label, indent
        self.detail, self.ok = "", True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def fail(self, message: str = "") -> None:
        self.ok, self.detail = False, message

    def _spin(self) -> None:
        i = 0
        while not self._stop.wait(0.12):
            frame = SYM["spin"][i % len(SYM["spin"])]
            line = f"{' ' * self.indent}{paint(frame, 'cyan')} {self.label}  {paint(f'{time.time() - self._t0:.0f}s', 'dim')}"
            _REAL_STDOUT.write("\r" + line)
            _REAL_STDOUT.flush()
            i += 1

    def __enter__(self) -> "Task":
        self._t0 = time.time()
        if _IS_TTY:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            _REAL_STDOUT.write(f"{' ' * self.indent}{SYM['arrow']} {self.label} ...\n")   # not into the log file
            _REAL_STDOUT.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            _REAL_STDOUT.write("\r" + " " * (term_width() + 10) + "\r")
            _REAL_STDOUT.flush()
        ok = self.ok and exc_type is None
        if exc_type is not None:
            self.detail = "interrupted" if exc_type is KeyboardInterrupt else f"{exc_type.__name__}: {exc}"
        mark = paint(SYM["ok"], "green", "bold") if ok else paint(SYM["fail"], "red", "bold")
        detail = f"  {self.detail}" if self.detail else ""
        say(f"{mark} {self.label}{detail}  {paint(f'({time.time() - self._t0:.1f} s)', 'dim')}", self.indent)
        return False

# ======================================================================
# CONSTANTS
# ======================================================================
DATASET_ID = "ds005385"
DATASET_VERSION = "1.0.3"
REPO_URL = f"https://github.com/OpenNeuroDatasets/{DATASET_ID}.git"
HTTPS_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"   # fallback only (always the newest snapshot)

TASKS = ["EyesClosed", "EyesOpen"]
ACQS = ["pre", "post"]
AGE_MAX = 60                      # inclusion rule 1: age <= 60

# ----------------------------------------------------------------------
# CONFIG: ALL QC thresholds live here (identical to the pilot notebook).
# The aim is a general quality check, not a strict one: a recording fails only when something is clearly wrong.
# ----------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    # ---- 1) header and metadata ----
    "expected_n_eeg_channels": 64,       # standard 64-channel layout (auxiliary channels such as Status/EOG/ECG are allowed)
    "expected_sfreq_hz": 1000.0,         # expected sampling rate
    "duration_ok_range_s": (120.0, 240.0),   # 2 to 4 minutes is accepted ("about 3 minutes"); outside = FLAG
    "duration_fail_below_s": 60.0,       # shorter than 1 minute = FAIL
    "required_channels": ["O1", "O2", "Oz"],  # occipital channels needed for the alpha check (missing = metadata FAIL)
    "non_eeg_name_hints": ["EOG", "VEOG", "HEOG", "ECG", "EKG", "EMG", "TRIG", "STI", "STATUS"],  # report only
    "edf_min_size_mb": 1.0,              # a real EDF is far bigger; a tiny file is a broken link

    # ---- 2a) baseline (drift) removal, applied IN MEMORY ONLY before the voltage statistics ----
    # Sweat / skin impedance cause slow DC drift. Amplitude, variance, skewness, kurtosis and the spatial
    # correlation are therefore computed on a baseline-corrected copy of the signal. Digital properties
    # (flat, clipping, frozen samples) and the DC-offset/drift numbers themselves are measured on the RAW signal.
    "detrend_method": "highpass",        # "highpass" (zero-phase Butterworth) | "linear" | "none"
    "detrend_hp_hz": 0.5,                # high-pass corner frequency
    "detrend_hp_order": 2,               # filter order (applied forward and backward)

    # ---- 2b) channel integrity ----
    "flat_std_uv": 0.1,                  # STD below 0.1 uV = flat channel (variance close to zero)
    "clip_fraction": 0.001,              # more than 0.1% of samples exactly at the channel min/max = clipping (saturation)
    "frozen_min_samples": 100,           # all channels unchanged for 100 samples (0.1 s) = "frozen" segment
    "noisy_ratio": 4.0,                  # a channel is "noisy" if its typical 1-s STD is 4x the median of all channels ...
    "noisy_min_uv": 50.0,                # ... and also above 50 uV
    "global_frozen_flag_s": 5.0,         # more than 5 s with ALL channels frozen = FLAG
    "global_frozen_fail_fraction": 0.10, # more than 10% of the recording frozen = FAIL

    # ---- 2c) spatial correlation (channels that are neither flat nor clipped, but unrelated to all the others) ----
    "corr_threshold": 0.25,              # |Pearson r| with the median of all channels below this = uncorrelated / noisy channel
    "corr_window_s": 5.0,                # r is computed per 5-s window; the channel's value is the median over windows
    "corr_band_hz": (1.0, 45.0),         # band used for the correlation (removes drift and the 50 Hz line)
    "corr_target_fs_hz": 125.0,          # the band-passed signal is down-sampled to about this rate (speed)
    "corr_unreliable_fraction": 0.5,     # if MORE than half of the channels fall below the threshold the reference is
                                         # not trusted (e.g. unusual referencing): nothing is marked bad, only a FLAG

    # ---- 2d) which bad channels can be repaired? (decision logic) ----
    "critical_channels": ["Cz", "Fz", "Pz", "O1", "O2"],   # the most important channels
    "critical_bad_fail_min": 2,          # 2 or more critical channels bad (e.g. both O1 and O2) = FAIL
    "interpolatable_max_channels": 6,    # 1..6 bad channels (none, or only one, critical) = PASS "USABLE_NEEDS_INTERPOLATION"
                                         # (more than 6 = too many to interpolate reliably = FAIL)

    # ---- 3) amplitude (visual trace inspection), on the baseline-corrected signal ----
    "typical_amp_min_uv": 1.0,           # typical 1-s STD below 1 uV = probable unit/scale problem (FAIL)
    "typical_amp_max_uv": 100.0,         # typical 1-s STD above 100 uV = probable unit/scale problem (FAIL)
    "high_amp_channel_fraction": 0.10,   # more than 10% of channels above the upper limit = FLAG
    # DC offset and drift are measured on the raw signal and reported, but they never change the verdict.
    "dc_offset_info_uv": 100.0,          # info only: |median| above this is listed as "DC offset"
    "drift_info_uv": 50.0,               # info only: first-minute vs last-minute median shift (also "orange" in the topomap)

    # ---- 4) spectral validation (Welch PSD) ----
    "psd_channels": ["O1", "O2", "Oz"],
    "welch_window_s": 4.0,               # 4 s Hann window (0.25 Hz resolution)
    "welch_overlap": 0.5,                # 50% overlap
    "alpha_band_hz": (8.0, 12.0),
    "alpha_flank_hz": ((4.0, 7.0), (13.0, 16.0)),   # flanks used to measure how much the alpha peak stands out
    # Alpha is only a FLAG: 10-15% of healthy people are "alpha non-responders" (no clear alpha peak, weak blocking).
    "alpha_prominence_min_db": 3.0,      # Eyes Closed: peak should stand out by at least 3 dB (else FLAG LOW_ALPHA_PROMINENCE)
    "alpha_ratio_pass": 1.5,             # alpha power Eyes Closed / Eyes Open: median >= 1.5 = nothing to note
    "alpha_no_blocking_ratio": 1.0,      # ratio below 1.0 on ALL occipital channels = FLAG NO_ALPHA_BLOCKING (not FAIL)
    # The only spectral FAIL: the occipital spectrum is dead (no signal power at all in the whole band).
    "dead_spectrum_band_hz": (1.0, 45.0),
    "dead_spectrum_power_uv2": 0.01,     # total power in the band below 0.01 uV^2 (= 0.1 uV RMS) on ALL of O1/O2/Oz
    "line_hz": 50.0,                     # European mains frequency
    "harmonics_hz": (100.0, 150.0),      # harmonics (reported only, to plan a notch filter)
    "line_flank_offsets_hz": (2.0, 5.0), # flanks of a line at f0: f0-5 .. f0-2 Hz and f0+2 .. f0+5 Hz (45-48 and 52-55 for 50 Hz)
    "line_peak_halfwidth_hz": 0.5,       # the line peak is searched in f0 +/- 0.5 Hz
    "line_flag_db": 20.0,                # 50 Hz more than 20 dB above its flanks = FLAG LINE_NOISE
    "line_high_db": 30.0,                # more than 30 dB = FLAG HIGH_LINE_NOISE (a notch filter removes it; never a FAIL by itself)
    "line_saturation_channel_fraction": 0.5,   # FAIL only if high line noise comes WITH clipping on at least this share of channels
    "notch_suggested_db": 10.0,          # info only: above 10 dB a notch filter is suggested for preprocessing
}

QC_LOGIC_VERSION = "screening-2.1"        # change this text whenever the QC logic changes (invalidates old results)


def qc_signature(dataset_version: str) -> str:
    """Short fingerprint of the QC logic + thresholds + dataset version.

    Results of an earlier run are only reused when this fingerprint is identical (same logic, same thresholds)."""
    blob = json.dumps({"logic": QC_LOGIC_VERSION, "config": CONFIG, "age_max": AGE_MAX, "dataset": dataset_version},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ======================================================================
# PARTICIPANT TABLE (the first 8 columns of the sheet; 608 participants).
# The final decision of the sheet is NOT stored: it is recomputed by check_eligibility().
# Use  --verify-table participants.tsv  to compare this table with your original file.
# ======================================================================
PARTICIPANT_COLUMNS = ["participant_id", "sex", "age", "handedness", "session1", "late_ses1", "session2", "late_ses2"]
N_PARTICIPANTS = 608

_PARTICIPANTS_TABLE = """\
participant_id sex age handedness session1 late_ses1 session2 late_ses2
sub-001 F 60 right yes 0 yes 0
sub-002 M 67 right yes 0 no n/a
sub-003 F 44 right yes 0 no n/a
sub-004 F 24 right yes 0 no n/a
sub-005 F 48 right yes 0 yes 5
sub-006 F 57 right yes 0 no n/a
sub-007 F 24 right yes 0 no n/a
sub-008 M 39 left yes 1901 no n/a
sub-009 F 37 right yes 2080 no n/a
sub-010 F 23 right yes 0 no n/a
sub-011 F 50 right yes 0 no n/a
sub-012 M 33 right yes 0 no n/a
sub-013 F 49 right yes 1364 no n/a
sub-014 M 64 right yes 0 no n/a
sub-015 F 61 right yes 0 no n/a
sub-016 M 32 right yes 0 no n/a
sub-017 M 50 right yes 0 no n/a
sub-018 F 20 right yes 0 no n/a
sub-019 F 42 left yes 0 no n/a
sub-020 M 68 right yes 0 no n/a
sub-021 F 23 left yes 0 no n/a
sub-022 M 60 right yes 0 no n/a
sub-023 F 31 right yes 0 no n/a
sub-024 F 47 right yes 0 no n/a
sub-025 F 26 left yes 0 no n/a
sub-026 F 59 right yes 0 no n/a
sub-027 F 51 right yes 15 no n/a
sub-028 M 48 right yes 0 yes 0
sub-029 F 36 right yes 0 no n/a
sub-030 F 62 right yes 0 no n/a
sub-031 F 26 right yes 0 no n/a
sub-032 F 20 right yes 0 no n/a
sub-033 M 32 right yes 0 yes 0
sub-034 M 41 right yes 0 no n/a
sub-035 M 36 right yes 0 no n/a
sub-036 F 53 left yes 0 no n/a
sub-037 M 30 right yes 445 no n/a
sub-038 M 49 right yes 0 yes 0
sub-039 M 55 right yes 0 no n/a
sub-040 M 48 right yes 0 no n/a
sub-041 M 53 right yes 0 yes 0
sub-042 F 22 right yes 0 no n/a
sub-043 M 40 left yes 0 no n/a
sub-044 F 27 right yes 0 yes 0
sub-045 F 47 right yes 0 no n/a
sub-046 F 49 right yes 0 yes 0
sub-047 M 67 right yes 0 yes 0
sub-048 M 53 right yes 0 yes 0
sub-049 F 67 right yes 34 yes 0
sub-050 F 59 right yes 0 yes 0
sub-051 F 58 right yes 0 yes 0
sub-052 F 59 right yes 1 yes 0
sub-053 F 42 right yes 0 yes 0
sub-054 M 31 right yes 0 yes 0
sub-055 M 50 right yes 35 no n/a
sub-056 F 52 right yes 0 no n/a
sub-057 F 67 right yes 0 no n/a
sub-058 F 62 right yes 0 no n/a
sub-059 M 60 right yes 0 yes 0
sub-060 F 51 right yes 0 yes 0
sub-061 M 46 right yes 0 no n/a
sub-062 M 31 right yes 0 no n/a
sub-063 M 27 right yes 0 no n/a
sub-064 M 51 right yes 0 no n/a
sub-065 F 50 right yes 0 no n/a
sub-066 F 33 right yes 0 no n/a
sub-067 F 20 right yes 0 no n/a
sub-068 M 42 right yes 0 no n/a
sub-069 M 47 right yes 0 no n/a
sub-070 M 70 right yes 0 no n/a
sub-071 F 64 right yes 0 no n/a
sub-072 F 33 right yes 0 no n/a
sub-073 F 35 right yes 0 yes 0
sub-074 F 56 n/a yes 0 no n/a
sub-075 F 56 right yes 1 yes 0
sub-076 M 56 right yes 0 yes 0
sub-077 M 64 right yes 0 yes 0
sub-078 F 50 right yes 0 no n/a
sub-079 F 33 right yes 0 no n/a
sub-080 F 64 left yes 0 yes 0
sub-081 M 48 right yes 0 no n/a
sub-082 F 26 right yes 0 no n/a
sub-083 F 62 right yes 0 no n/a
sub-084 F 28 right yes 0 no n/a
sub-085 F 22 right yes 99 no n/a
sub-086 F 31 right yes 0 yes 0
sub-087 F 51 right yes 0 no n/a
sub-088 F 40 right yes 0 no n/a
sub-089 M 31 right yes 0 no n/a
sub-090 F 60 right yes 0 no n/a
sub-091 M 27 right yes 0 yes 0
sub-092 F 40 right yes 0 yes 0
sub-093 F 54 right yes 14 no n/a
sub-094 F 51 right yes 0 no n/a
sub-095 F 54 right yes 0 no n/a
sub-096 F 27 right yes 0 yes 0
sub-097 F 53 right yes 50 no n/a
sub-098 F 59 right yes 0 yes 0
sub-099 F 20 right yes 0 no n/a
sub-100 M 65 right yes 0 yes 0
sub-101 F 49 left yes 0 no n/a
sub-102 F 28 right yes 0 no n/a
sub-103 M 23 right yes 0 no n/a
sub-104 M 63 right yes 0 no n/a
sub-105 F 22 right yes 0 yes 0
sub-106 F 66 right yes 9 no n/a
sub-107 M 35 right yes 0 no n/a
sub-108 F 57 right yes 22 no n/a
sub-109 F 44 right yes 0 yes 0
sub-110 M 26 right yes 0 yes 0
sub-111 M 67 right yes 0 yes 0
sub-112 F 70 right yes 0 yes 0
sub-113 M 40 right yes 0 no n/a
sub-114 M 27 right yes 1 no n/a
sub-115 F 59 right yes 1 no n/a
sub-116 F 37 right yes 0 no n/a
sub-117 M 59 left yes 0 no n/a
sub-118 F 30 right yes 0 no n/a
sub-119 F 54 right yes 0 yes 0
sub-120 M 70 right yes 0 no n/a
sub-121 F 21 right yes 0 no n/a
sub-122 M 48 right yes 0 no n/a
sub-123 F 50 right yes 0 yes 0
sub-124 M 62 right yes 0 yes 0
sub-125 M 70 right yes 0 no n/a
sub-126 F 70 right yes 0 no n/a
sub-127 F 45 right yes 0 no n/a
sub-128 M 30 right yes 0 no n/a
sub-129 F 40 left yes 0 yes 0
sub-130 M 25 right yes 0 no n/a
sub-131 F 65 right yes 0 yes 0
sub-132 M 56 right yes 0 no n/a
sub-133 F 68 right yes 0 no n/a
sub-134 F 24 right yes 0 no n/a
sub-135 M 53 right yes 0 yes 0
sub-136 F 24 right yes 0 no n/a
sub-137 M 35 right yes 0 yes 0
sub-138 F 50 right yes 366 yes 0
sub-139 M 67 right yes 0 yes 0
sub-140 F 46 right yes 0 no n/a
sub-141 F 61 right yes 0 no n/a
sub-142 F 23 right yes 0 yes 2
sub-143 M 29 right yes 0 yes 0
sub-144 F 25 right yes 0 no n/a
sub-145 M 60 right yes 0 yes 0
sub-146 F 33 right yes 0 no n/a
sub-147 M 40 right yes 0 yes 0
sub-148 F 25 right yes 0 no n/a
sub-149 F 21 right yes 0 no n/a
sub-150 F 40 right yes 0 no n/a
sub-151 M 53 left yes 0 no n/a
sub-152 M 54 right yes 0 no n/a
sub-153 F 29 right yes 0 no n/a
sub-154 M 51 right yes 0 no n/a
sub-155 M 62 right yes 0 no n/a
sub-156 M 21 right yes 0 no n/a
sub-157 F 57 right yes 0 no n/a
sub-158 F 35 right yes 0 no n/a
sub-159 F 29 right yes 0 no n/a
sub-160 M 57 right yes 0 yes 0
sub-161 F 34 right yes 0 no n/a
sub-162 M 53 left yes 0 no n/a
sub-163 F 20 right yes 0 no n/a
sub-164 F 67 right yes 0 yes 0
sub-165 F 49 right yes 0 yes 0
sub-166 M 57 right yes 0 no n/a
sub-167 M 60 right yes 0 no n/a
sub-168 F 50 right yes 0 yes 0
sub-169 F 63 right yes 0 no n/a
sub-170 F 59 right yes 42 yes 0
sub-171 F 59 right yes 0 yes 0
sub-172 F 31 right yes 0 yes 0
sub-173 F 33 left yes 0 no n/a
sub-174 M 34 right yes 0 no n/a
sub-175 F 25 right yes 0 no n/a
sub-176 F 62 right yes 16 no n/a
sub-177 F 45 right yes 0 no n/a
sub-178 F 30 right yes 0 no n/a
sub-179 F 21 right yes 1884 no n/a
sub-180 F 33 right yes 0 no n/a
sub-181 M 37 right yes 0 yes 0
sub-182 F 63 right yes 0 no n/a
sub-183 F 53 right yes 29 yes 0
sub-184 F 29 right yes 0 yes 0
sub-185 M 31 right yes 0 no n/a
sub-186 F 61 right yes 0 no n/a
sub-187 F 20 right yes 0 no n/a
sub-188 F 43 right yes 0 yes 0
sub-189 F 24 right yes 0 yes 0
sub-190 M 48 right yes 0 yes 0
sub-191 F 59 right yes 0 no n/a
sub-192 F 26 right yes 0 no n/a
sub-193 F 31 right yes 0 no n/a
sub-194 F 46 right yes 0 no n/a
sub-195 F 60 right yes 0 yes 0
sub-196 F 47 right yes 0 no n/a
sub-197 M 57 left yes 0 no n/a
sub-198 F 48 right yes 0 yes 0
sub-199 F 31 right yes 0 no n/a
sub-200 F 50 right yes 0 no n/a
sub-201 F 37 right yes 0 no n/a
sub-202 M 26 right yes 0 no n/a
sub-203 F 67 right yes 0 no n/a
sub-204 F 23 right yes 0 no n/a
sub-205 F 65 right yes 0 no n/a
sub-206 M 33 right yes 9 no n/a
sub-207 F 53 right yes 916 no n/a
sub-208 M 23 right yes 0 no n/a
sub-209 F 20 right yes 0 no n/a
sub-210 M 42 right yes 0 yes 0
sub-211 F 58 left yes 0 yes 0
sub-212 F 45 right yes 0 no n/a
sub-213 M 66 right yes 0 no n/a
sub-214 M 44 right yes 0 no n/a
sub-215 M 56 right yes 0 no n/a
sub-216 M 39 right yes 0 yes 0
sub-217 F 36 right yes 0 yes 0
sub-218 F 21 right yes 0 no n/a
sub-219 M 20 left yes 0 no n/a
sub-220 M 50 right yes 1838 no n/a
sub-221 F 24 right yes 0 no n/a
sub-222 M 33 right yes 89 yes 0
sub-223 F 26 left yes 0 no n/a
sub-224 M 22 right yes 0 no n/a
sub-225 F 43 right yes 0 yes 0
sub-226 F 56 right yes 1947 yes 0
sub-227 M 58 right yes 0 yes 0
sub-228 F 58 right yes 0 no n/a
sub-229 F 46 right yes 0 no n/a
sub-230 F 26 right yes 0 yes 0
sub-231 M 40 left yes 0 yes 0
sub-232 F 38 right yes 0 no n/a
sub-233 F 29 right yes 440 no n/a
sub-234 F 68 right yes 0 yes 1
sub-235 F 27 right yes 0 yes 0
sub-236 F 57 right yes 0 no n/a
sub-237 M 26 right yes 1 no n/a
sub-238 F 35 left yes 0 yes 0
sub-239 F 42 right yes 0 yes 0
sub-240 F 34 right yes 0 no n/a
sub-241 F 61 right yes 0 yes 0
sub-242 M 59 right yes 0 yes 0
sub-243 F 27 left yes 0 no n/a
sub-244 F 65 right yes 0 no n/a
sub-245 F 56 right yes 1915 yes 0
sub-246 F 64 right yes 0 no n/a
sub-247 F 41 left yes 0 no n/a
sub-248 F 26 right yes 0 yes 0
sub-249 F 53 right yes 0 no n/a
sub-250 F 54 right yes 0 no n/a
sub-251 F 51 left yes 0 no n/a
sub-252 F 51 right yes 0 yes 0
sub-253 M 29 right yes 1 no n/a
sub-254 M 52 right yes 6 no n/a
sub-255 M 30 right yes 0 no n/a
sub-256 F 45 right yes 0 yes 0
sub-257 F 49 right yes 0 no n/a
sub-258 M 33 right yes 0 no n/a
sub-259 F 46 right yes 0 no n/a
sub-260 F 32 right yes 0 yes 3
sub-261 F 29 right yes 0 no n/a
sub-262 M 50 right yes 0 yes 0
sub-263 F 41 right yes 0 yes 0
sub-264 F 26 right yes 0 no n/a
sub-265 M 62 right yes 0 yes 0
sub-266 F 44 right yes 0 yes 0
sub-267 M 26 right yes 0 yes 0
sub-268 F 53 right yes 0 no n/a
sub-269 M 52 right yes 0 no n/a
sub-270 M 47 right yes 2075 yes 0
sub-271 M 66 right yes 0 no n/a
sub-272 F 39 right yes 0 yes 0
sub-273 F 26 right yes 0 no n/a
sub-274 F 26 right yes 0 no n/a
sub-275 F 22 right yes 0 no n/a
sub-276 F 58 right yes 0 no n/a
sub-277 F 32 right yes 0 yes 0
sub-278 M 26 left yes 195 no n/a
sub-279 F 40 right yes 2023 no n/a
sub-280 M 50 right yes 0 yes 0
sub-281 M 31 right yes 0 no n/a
sub-282 F 52 right yes 0 yes 0
sub-283 F 64 right yes 0 no n/a
sub-284 F 34 right yes 0 no n/a
sub-285 M 46 right yes 0 no n/a
sub-286 F 54 right yes 16 no n/a
sub-287 M 68 right yes 0 no n/a
sub-288 F 57 right yes 0 yes 0
sub-289 M 24 right yes 0 no n/a
sub-290 M 29 right yes 0 no n/a
sub-291 F 50 right yes 0 no n/a
sub-292 F 57 right yes 0 no n/a
sub-293 M 35 right yes 32 no n/a
sub-294 F 48 right yes 0 yes 0
sub-295 F 65 right yes 0 yes 0
sub-296 F 30 right yes 125 yes 0
sub-297 M 38 right yes 0 no n/a
sub-298 M 49 right yes 0 no n/a
sub-299 M 35 right yes 0 yes 0
sub-300 F 45 right yes 0 yes 0
sub-301 M 40 right yes 0 no n/a
sub-302 F 50 left yes 0 no n/a
sub-303 F 30 left yes 0 no n/a
sub-304 F 50 right yes 0 no n/a
sub-305 M 42 right yes 0 yes 0
sub-306 F 54 right yes 2208 no n/a
sub-307 M 37 right yes 0 no n/a
sub-308 M 25 right yes 0 yes 0
sub-309 M 60 right yes 0 no n/a
sub-310 F 52 right yes 0 no n/a
sub-311 F 46 left yes 0 no n/a
sub-312 F 65 right yes 0 no n/a
sub-313 F 51 right yes 0 no n/a
sub-314 F 30 right yes 1 yes 0
sub-315 F 57 right yes 0 yes 0
sub-316 M 62 right yes 0 yes 0
sub-317 F 70 right yes 23 yes 0
sub-318 F 67 right yes 0 no n/a
sub-319 M 49 right yes 1 no n/a
sub-320 F 62 left yes 0 no n/a
sub-321 F 51 right yes 0 no n/a
sub-322 F 41 right yes 1 yes 0
sub-323 M 56 right yes 0 yes 0
sub-324 M 27 right yes 20 no n/a
sub-325 M 24 right yes 0 no n/a
sub-326 F 43 right yes 0 yes 0
sub-327 M 31 right yes 25 no n/a
sub-328 F 48 right yes 0 no n/a
sub-329 M 68 right yes 876 yes 0
sub-330 F 29 right yes 0 no n/a
sub-331 F 64 right yes 0 no n/a
sub-332 F 27 right yes 0 yes 0
sub-333 M 39 right yes 0 no n/a
sub-334 F 58 right yes 0 yes 0
sub-335 M 40 right yes 1 no n/a
sub-336 M 39 right yes 11 no n/a
sub-337 M 30 right yes 0 no n/a
sub-338 M 30 right yes 0 no n/a
sub-339 F 65 left yes 0 no n/a
sub-340 F 52 right yes 0 yes 0
sub-341 M 51 right yes 0 yes 0
sub-342 M 26 right yes 1 no n/a
sub-343 F 29 right yes 0 no n/a
sub-344 M 51 right yes 0 no n/a
sub-345 F 27 right yes 0 no n/a
sub-346 M 60 left yes 0 no n/a
sub-347 M 29 right yes 0 yes 0
sub-348 F 32 right yes 0 yes 0
sub-349 F 65 right yes 0 yes 0
sub-350 F 36 right yes 0 no n/a
sub-351 F 47 right yes 0 yes 0
sub-352 F 28 right yes 0 yes 0
sub-353 F 24 right yes 0 yes 0
sub-354 F 25 right yes 0 no n/a
sub-355 F 56 right yes 0 no n/a
sub-356 M 49 right yes 38 no n/a
sub-357 F 49 right yes 1875 yes 0
sub-358 F 23 right yes 0 no n/a
sub-359 F 70 right yes 2370 no n/a
sub-360 M 54 right yes 43 no n/a
sub-361 M 36 right yes 0 no n/a
sub-362 M 25 right yes 1617 yes 0
sub-363 F 23 right yes 0 no n/a
sub-364 M 34 left yes 0 no n/a
sub-365 M 33 right yes 0 yes 0
sub-366 F 50 right yes 0 no n/a
sub-367 F 24 right yes 0 yes 0
sub-368 F 60 right yes 0 yes 0
sub-369 F 26 right yes 211 no n/a
sub-370 F 66 right yes 0 no n/a
sub-371 M 23 right yes 0 no n/a
sub-372 M 57 right yes 1888 yes 0
sub-373 F 32 right yes 0 no n/a
sub-374 F 27 right yes 0 yes 0
sub-375 F 46 right yes 0 no n/a
sub-376 M 60 right yes 0 no n/a
sub-377 F 47 right yes 0 yes 0
sub-378 M 37 right yes 0 no n/a
sub-379 F 56 right yes 0 no n/a
sub-380 F 55 right yes 5 no n/a
sub-381 M 27 right yes 0 no n/a
sub-382 M 64 right yes 0 yes 0
sub-383 F 48 right yes 0 no n/a
sub-384 M 55 right yes 1869 no n/a
sub-385 F 41 left yes 0 yes 0
sub-386 M 60 right yes 0 no n/a
sub-387 F 20 right yes 0 no n/a
sub-388 F 22 right yes 0 no n/a
sub-389 M 67 right yes 0 no n/a
sub-390 F 36 right yes 0 no n/a
sub-391 M 49 right yes 0 yes 0
sub-392 M 26 left yes 66 no n/a
sub-393 M 59 right yes 2049 yes 0
sub-394 F 49 right yes 0 yes 0
sub-395 M 45 right yes 0 yes 0
sub-396 F 43 right yes 0 no n/a
sub-397 F 25 right yes 0 no n/a
sub-398 M 53 right yes 1950 yes 0
sub-399 M 55 right yes 0 no n/a
sub-400 F 46 right yes 0 yes 0
sub-401 M 56 right yes 0 no n/a
sub-402 F 50 right yes 0 no n/a
sub-403 F 60 right yes 0 yes 19
sub-404 F 47 right yes 0 yes 0
sub-405 F 52 right yes 0 no n/a
sub-406 F 58 right yes 0 no n/a
sub-407 F 20 right yes 0 no n/a
sub-408 F 23 right yes 0 no n/a
sub-409 M 37 right yes 0 no n/a
sub-410 M 37 right yes 0 yes 0
sub-411 F 32 right yes 0 yes 0
sub-412 M 68 right yes 0 no n/a
sub-413 M 35 right yes 0 yes 0
sub-414 F 54 right yes 0 no n/a
sub-415 F 62 right yes 1 yes 1
sub-416 F 28 right yes 0 no n/a
sub-417 F 25 right yes 0 yes 0
sub-418 F 46 right yes 0 yes 0
sub-419 F 52 right yes 0 no n/a
sub-420 F 53 right yes 0 no n/a
sub-421 F 37 right yes 0 yes 0
sub-422 F 25 right yes 0 no n/a
sub-423 F 51 right yes 0 yes 0
sub-424 M 25 right yes 0 no n/a
sub-425 F 49 right yes 15 yes 0
sub-426 M 30 right yes 0 yes 0
sub-427 F 22 right yes 0 no n/a
sub-428 F 55 right yes 0 no n/a
sub-429 F 24 right yes 0 no n/a
sub-430 F 48 right yes 43 no n/a
sub-431 F 23 right yes 0 no n/a
sub-432 M 29 right yes 1929 no n/a
sub-433 F 67 left yes 0 no n/a
sub-434 F 36 right yes 0 no n/a
sub-435 F 35 right yes 0 no n/a
sub-436 F 26 right yes 0 yes 0
sub-437 M 31 right yes 0 no n/a
sub-438 M 70 right yes 0 yes 0
sub-439 M 23 right yes 0 no n/a
sub-440 M 35 right yes 0 no n/a
sub-441 F 27 right yes 1918 yes 136
sub-442 F 63 right yes 0 no n/a
sub-443 F 27 right yes 0 no n/a
sub-444 M 42 right yes 0 no n/a
sub-445 M 46 right yes 0 yes 0
sub-446 F 20 right yes 0 yes 0
sub-447 F 61 right yes 0 no n/a
sub-448 F 58 right yes 0 no n/a
sub-449 M 23 left yes 0 no n/a
sub-450 F 47 right yes 0 no n/a
sub-451 F 27 left yes 0 no n/a
sub-452 M 41 right yes 0 no n/a
sub-453 M 34 right yes 0 no n/a
sub-454 M 69 right yes 0 no n/a
sub-455 F 63 right yes 0 yes 0
sub-456 F 21 right yes 0 no n/a
sub-457 M 42 right yes 0 no n/a
sub-458 M 31 right yes 0 no n/a
sub-459 F 32 right yes 0 no n/a
sub-460 M 60 right yes 0 no n/a
sub-461 F 41 right yes 0 no n/a
sub-462 M 58 right yes 0 no n/a
sub-463 F 37 right yes 0 yes 0
sub-464 F 50 right yes 0 no n/a
sub-465 F 48 right yes 0 yes 0
sub-466 M 59 right yes 0 yes 0
sub-467 F 33 right yes 0 yes 0
sub-468 F 54 right yes 0 no n/a
sub-469 F 32 right yes 0 no n/a
sub-470 M 70 right yes 0 no n/a
sub-471 M 29 right yes 2 no n/a
sub-472 M 37 right yes 2 no n/a
sub-473 M 62 right yes 0 yes 0
sub-474 M 29 right yes 2100 no n/a
sub-475 M 47 right yes 0 yes 0
sub-476 F 52 right yes 0 yes 0
sub-477 F 43 right yes 1 no n/a
sub-478 F 42 right yes 0 yes 0
sub-479 F 39 right yes 0 no n/a
sub-480 F 57 right yes 0 yes 0
sub-481 F 70 right yes 0 yes 0
sub-482 F 32 right yes 0 no n/a
sub-483 M 30 right yes 0 no n/a
sub-484 F 53 right yes 1867 no n/a
sub-485 F 36 right yes 0 yes 0
sub-486 M 69 right yes 1 no n/a
sub-487 M 55 right yes 0 yes 0
sub-488 F 25 right yes 0 no n/a
sub-489 F 58 right yes 0 yes 0
sub-490 M 56 right yes 13 yes 0
sub-491 F 51 right yes 1 no n/a
sub-492 F 29 right yes 0 no n/a
sub-493 F 24 right yes 1 no n/a
sub-494 M 59 right yes 2 no n/a
sub-495 F 36 right yes 0 no n/a
sub-496 F 33 right yes 0 yes 0
sub-497 M 29 right yes 0 yes 0
sub-498 F 56 right yes 0 yes 0
sub-499 F 58 right yes 0 no n/a
sub-500 M 32 right yes 0 no n/a
sub-501 F 49 right yes 0 no n/a
sub-502 F 38 right yes 0 yes 0
sub-503 F 48 right yes 0 yes 0
sub-504 F 40 left yes 0 no n/a
sub-505 F 62 right yes 0 yes 0
sub-506 M 30 right yes 1 no n/a
sub-507 F 57 right yes 0 no n/a
sub-508 F 54 right yes 0 no n/a
sub-509 M 69 right yes 0 yes 1
sub-510 F 52 right yes 0 yes 0
sub-511 M 66 right yes 0 yes 0
sub-512 F 40 right yes 0 yes 0
sub-513 F 58 right yes 0 no n/a
sub-514 F 48 right yes 0 no n/a
sub-515 F 26 right yes 0 no n/a
sub-516 M 30 right yes 0 yes 0
sub-517 M 66 left yes 0 yes 0
sub-518 M 48 n/a yes 0 no n/a
sub-519 F 48 right yes 0 no n/a
sub-520 M 70 right yes 0 yes 0
sub-521 M 36 right yes 0 no n/a
sub-522 F 24 right yes 0 yes 0
sub-523 F 62 right yes 0 no n/a
sub-524 M 47 right yes 0 no n/a
sub-525 F 44 right yes 0 no n/a
sub-526 M 28 left yes 1 no n/a
sub-527 M 52 right yes 0 no n/a
sub-528 M 56 right yes 0 no n/a
sub-529 M 20 right yes 0 no n/a
sub-530 F 26 right yes 0 yes 0
sub-531 F 45 right yes 0 no n/a
sub-532 F 30 right yes 0 no n/a
sub-533 F 36 right yes 0 yes 0
sub-534 M 43 right yes 0 yes 0
sub-535 F 31 right yes 20 yes 0
sub-536 F 24 right yes 0 yes 0
sub-537 M 61 right yes 0 no n/a
sub-538 F 48 right yes 0 no n/a
sub-539 F 27 right yes 0 no n/a
sub-540 F 32 right yes 0 no n/a
sub-541 F 45 right yes 22 no n/a
sub-542 F 25 right yes 376 no n/a
sub-543 M 57 right yes 0 yes 0
sub-544 F 32 right yes 0 no n/a
sub-545 M 47 right yes 0 yes 0
sub-546 F 53 right yes 0 yes 0
sub-547 F 53 right yes 1 no n/a
sub-548 M 46 right yes 0 no n/a
sub-549 M 53 right yes 0 yes 0
sub-550 M 63 right yes 0 yes 0
sub-551 F 62 right yes 0 no n/a
sub-552 F 53 right yes 0 yes 0
sub-553 F 23 right yes 0 no n/a
sub-554 F 29 right yes 1 no n/a
sub-555 F 70 right yes 0 yes 0
sub-556 M 60 right yes 0 no n/a
sub-557 M 64 right yes 0 no n/a
sub-558 M 70 right yes 0 no n/a
sub-559 M 61 right yes 0 yes 0
sub-560 F 27 right yes 0 no n/a
sub-561 F 51 right yes 0 no n/a
sub-562 M 31 right yes 0 no n/a
sub-563 M 70 right yes 0 yes 0
sub-564 F 49 right yes 0 yes 1
sub-565 F 28 right yes 0 yes 0
sub-566 M 64 right yes 0 no n/a
sub-567 F 52 right yes 0 no n/a
sub-568 F 54 right yes 0 yes 0
sub-569 F 26 right yes 1 no n/a
sub-570 M 49 right yes 0 yes 0
sub-571 M 61 right yes 1 no n/a
sub-572 F 49 right yes 0 no n/a
sub-573 F 30 right yes 0 no n/a
sub-574 F 40 right yes 0 no n/a
sub-575 M 38 right yes 1906 yes 0
sub-576 M 54 right yes 0 yes 0
sub-577 F 51 right yes 1 no n/a
sub-578 M 24 right yes 0 no n/a
sub-579 F 24 right yes 111 no n/a
sub-580 F 20 right yes 0 no n/a
sub-581 M 37 right yes 0 no n/a
sub-582 F 58 right yes 1870 no n/a
sub-583 M 64 left yes 0 no n/a
sub-584 F 28 right yes 0 no n/a
sub-585 M 50 right yes 0 no n/a
sub-586 M 41 right yes 0 no n/a
sub-587 M 69 right yes 0 yes 0
sub-588 F 40 right yes 0 no n/a
sub-589 F 31 right yes 0 no n/a
sub-590 M 28 right yes 1188 no n/a
sub-591 F 49 right yes 0 yes 0
sub-592 M 68 right yes 0 no n/a
sub-593 F 26 right yes 0 yes 2
sub-594 F 24 left yes 0 no n/a
sub-595 M 65 right yes 0 no n/a
sub-596 F 45 right yes 14 no n/a
sub-597 F 31 right yes 0 no n/a
sub-598 M 33 right yes 0 no n/a
sub-599 F 67 right yes 0 yes 0
sub-600 F 35 right yes 0 no n/a
sub-601 F 37 right yes 0 no n/a
sub-602 F 63 right yes 34 yes 0
sub-603 M 55 right yes 0 no n/a
sub-604 M 29 right yes 0 no n/a
sub-605 M 51 right yes 0 no n/a
sub-606 F 61 right yes 0 yes 0
sub-607 M 66 right yes 0 no n/a
sub-608 F 25 right yes 0 yes 0
"""


def _parse_embedded_table(text: str) -> dict[str, dict[str, str]]:
    lines = [ln.split() for ln in text.strip().splitlines() if ln.strip()]
    header, rows = lines[0], lines[1:]
    if header != PARTICIPANT_COLUMNS:
        raise ValueError(f"Embedded table: unexpected header {header}")
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if len(r) != len(header):
            raise ValueError(f"Embedded table: bad row {r}")
        out[r[0]] = dict(zip(header[1:], r[1:]))
    expected_ids = [f"sub-{i:03d}" for i in range(1, N_PARTICIPANTS + 1)]
    if list(out) != expected_ids:
        raise ValueError("Embedded table is damaged: expected sub-001 ... sub-608 in order.")
    return out


PARTICIPANTS = _parse_embedded_table(_PARTICIPANTS_TABLE)


def _to_int(v: Any) -> int | None:
    """'12' -> 12, '7.0' -> 7, 'n/a' / '' / 'nan' -> None."""
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f != int(f):
        return None
    return int(f)


# ======================================================================
# ELIGIBILITY (checked BEFORE anything is downloaded)
# ======================================================================
@dataclass
class Eligibility:
    ok: bool
    sessions: list[str]               # sessions that will be analysed, e.g. ["ses-1"] or ["ses-1", "ses-2"]
    reasons: list[str] = field(default_factory=list)   # why the subject is rejected
    notes: list[str] = field(default_factory=list)     # information for an accepted subject
    rules: list[dict[str, str]] = field(default_factory=list)   # [{"rule", "result": PASS|FAIL|INFO, "detail"}]


def check_eligibility(row: dict[str, str]) -> Eligibility:
    """Rule 1: age <= 60.  Rule 3: no late triggers in any completed session.
    Rule 2 of the sheet (both sessions) is relaxed: a subject with session 1 only is accepted."""
    reasons: list[str] = []
    notes: list[str] = []
    rules: list[dict[str, str]] = []

    age = _to_int(row["age"])
    if age is None:
        reasons.append(f"Age not numeric ({row['age']})")
        rules.append({"rule": f"Age <= {AGE_MAX}", "result": "FAIL", "detail": f"age is not a number ({row['age']})"})
    elif age > AGE_MAX:
        reasons.append(f"Age above {AGE_MAX} ({age})")
        rules.append({"rule": f"Age <= {AGE_MAX}", "result": "FAIL", "detail": f"age {age} is above the limit"})
    else:
        rules.append({"rule": f"Age <= {AGE_MAX}", "result": "PASS", "detail": f"age {age}"})

    s1 = row["session1"].strip().lower() == "yes"
    s2 = row["session2"].strip().lower() == "yes"
    if not s1:
        reasons.append("Session 1 not completed")
    rules.append({"rule": "Session 1 completed", "result": "PASS" if s1 else "FAIL",
                  "detail": "yes" if s1 else "no"})

    l1, l2 = _to_int(row["late_ses1"]), _to_int(row["late_ses2"])
    late_detail = f"late_ses1={row['late_ses1']}, late_ses2={row['late_ses2']}"
    if (l1 is not None and l1 > 0) or (l2 is not None and l2 > 0):
        reasons.append(f"Late triggers ({late_detail})")
        rules.append({"rule": "No late triggers", "result": "FAIL", "detail": late_detail + "  (delayed triggers)"})
    elif (s1 and l1 is None) or (s2 and l2 is None):
        reasons.append(f"Late-trigger count missing for a completed session ({late_detail})")
        rules.append({"rule": "No late triggers", "result": "FAIL", "detail": late_detail + "  (count missing)"})
    else:
        rules.append({"rule": "No late triggers", "result": "PASS", "detail": late_detail})

    if s2:
        rules.append({"rule": "Session 2 completed", "result": "PASS", "detail": "yes"})
    else:
        rules.append({"rule": "Session 2 completed", "result": "INFO",
                      "detail": "no (allowed): only session 1 is analysed"})
        if s1:
            notes.append("Session 2 was not completed: only ses-1 (4 recordings) will be analysed.")

    sessions = (["ses-1"] if s1 else []) + (["ses-2"] if s2 else [])
    return Eligibility(ok=not reasons, sessions=sessions, reasons=reasons, notes=notes, rules=rules)


def _num_token(tok: str) -> int | None:
    t = tok.strip().lower()
    if t.startswith("sub-"):          # tolerated, but you do not need to type it
        t = t[4:]
    return int(t) if t.isdigit() else None


def parse_selection(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Turn what the user typed into subject ids.

        "18"            -> sub-018
        "1 2 3"         -> sub-001 sub-002 sub-003
        "1 to 20"       -> sub-001 ... sub-020
        "1 5 to 8 12"   -> sub-001 sub-005 sub-006 sub-007 sub-008 sub-012      (both styles can be mixed)

    Commas are accepted as spaces. Duplicates are dropped (the first position is kept).
    Returns (subject ids, list of error messages).
    """
    toks = " ".join(tokens).replace(",", " ").split()
    nums: list[int] = []
    errors: list[str] = []

    def in_range(n: int) -> bool:
        if 1 <= n <= N_PARTICIPANTS:
            return True
        errors.append(f"{n} is outside 1-{N_PARTICIPANTS}")
        return False

    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.lower() == "to":
            errors.append("'to' must stand between two numbers, for example:  1 to 20")
            i += 1
            continue
        a = _num_token(tok)
        if a is None:
            errors.append(f"'{tok}' is not a subject number")
            i += 1
            continue
        if i + 1 < len(toks) and toks[i + 1].lower() == "to":
            b = _num_token(toks[i + 2]) if i + 2 < len(toks) else None
            if b is None:
                errors.append(f"'{a} to' needs a number after 'to'")
                i += 2
                continue
            if a > b:
                errors.append(f"'{a} to {b}': the first number must not be larger than the second")
            elif in_range(a) and in_range(b):
                nums.extend(range(a, b + 1))
            i += 3
            continue
        if in_range(a):
            nums.append(a)
        i += 1

    seen: set[int] = set()
    subjects = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            subjects.append(f"sub-{n:03d}")
    return subjects, errors


def ask_subjects() -> list[str]:
    """Interactive question. Repeats up to 3 times when the answer is not valid."""
    if not sys.stdin or not sys.stdin.isatty():
        sys.exit("ERROR: no subjects given and no terminal available. Run e.g.  python qc_eeg_subject.py 1 to 20")
    print()
    say(paint(f"Which subjects do you want to check?  (numbers 1-{N_PARTICIPANTS})", "bold"))
    say("one subject      18", 4)
    say("several          1 2 3", 4)
    say("a range          1 to 20", 4)
    say("mixed            1 5 to 8 12", 4)
    while True:                      # asks again until the answer is valid (Ctrl+C or Ctrl+D to quit)
        try:
            answer = input(paint("\n  Subjects > ", "cyan", "bold"))
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit("Cancelled: no subjects were selected.")
        subjects, errors = parse_selection([answer])
        if subjects and not errors:
            return subjects
        for e in errors or ["nothing was entered"]:
            bad(e, 4)


# ======================================================================
# TABLE COMPARISON (--verify-table, and a cross-check with the dataset's participants.tsv)
# ======================================================================
def _norm(v: Any) -> str:
    s = str(v).strip().lower()
    i = _to_int(s)
    return str(i) if i is not None else s


def compare_with_tsv(tsv_path: Path, only_sub: str | None = None) -> tuple[int, list[str]]:
    """Compare the embedded table with a TSV. Returns (rows compared, list of differences)."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if "participant_id" not in df.columns:
        raise ValueError(f"{tsv_path}: no 'participant_id' column")
    cols = [c for c in PARTICIPANT_COLUMNS[1:] if c in df.columns]
    diffs: list[str] = []
    n = 0
    for _, r in df.iterrows():
        pid = str(r["participant_id"]).strip()
        if only_sub and pid != only_sub:
            continue
        n += 1
        emb = PARTICIPANTS.get(pid)
        if emb is None:
            diffs.append(f"{pid}: not in the embedded table")
            continue
        for c in cols:
            if _norm(r[c]) != _norm(emb[c]):
                diffs.append(f"{pid}: {c}: file={r[c]!r} embedded={emb[c]!r}")
    if not only_sub:
        for pid in sorted(set(PARTICIPANTS) - set(df["participant_id"].astype(str).str.strip())):
            diffs.append(f"{pid}: missing in {tsv_path.name}")
    return n, diffs


def verify_table(path: str) -> int:
    p = Path(path)
    if not p.is_file():
        print(f"ERROR: file not found: {p}")
        return 1
    try:
        n, diffs = compare_with_tsv(p)
    except Exception as e:
        print(f"ERROR: could not read {p}: {type(e).__name__}: {e}")
        return 1
    print(f"Compared {n} rows of {p} with the embedded table (columns: {', '.join(PARTICIPANT_COLUMNS[1:])}).")
    if diffs:
        print(f"{len(diffs)} difference(s):")
        for d in diffs[:50]:
            print("  ", d)
        if len(diffs) > 50:
            print(f"   ... and {len(diffs) - 50} more")
        return 1
    print("OK: the embedded table is identical to the file.")
    return 0


# ======================================================================
# FILE NAMES
# ======================================================================
def recording_files(sub: str, ses: str, task: str, acq: str) -> dict[str, str]:
    """Paths of ONE recording relative to the dataset root (BIDS)."""
    base = f"{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_acq-{acq}"
    return {"edf": f"{base}_eeg.edf", "json": f"{base}_eeg.json", "events": f"{base}_events.tsv"}


def recording_paths(data_dir: Path, sub: str, ses: str, task: str, acq: str) -> dict[str, Path]:
    return {k: data_dir / rel for k, rel in recording_files(sub, ses, task, acq).items()}


def wanted_files(sub: str, sessions: list[str]) -> list[tuple[str, str]]:
    """(kind, relative path) of every file of the subject that the QC can use: edf (required), json, events."""
    return [(kind, rel) for ses in sessions for task in TASKS for acq in ACQS
            for kind, rel in recording_files(sub, ses, task, acq).items()]


def edf_is_real(p: Path) -> bool:
    return p.exists() and p.stat().st_size >= CONFIG["edf_min_size_mb"] * 1e6


# ======================================================================
# DOWNLOAD
# ======================================================================
def _short_rec(rel: str) -> str:
    """'sub-018/ses-1/eeg/sub-018_ses-1_task-EyesClosed_acq-pre_eeg.edf' -> 'ses-1 EC pre'"""
    m = re.search(r"_(ses-\d+)_task-(\w+?)_acq-(\w+?)_eeg\.edf$", rel)
    return f"{m.group(1)} {'EC' if m.group(2) == 'EyesClosed' else 'EO'} {m.group(3)}" if m else Path(rel).name


def _warn_absent(absent: list[str]) -> None:
    if absent:
        warn(f"{len(absent)} recording(s) do not exist in the dataset (they will be reported as MISSING): "
             + ", ".join(_short_rec(r) for r in absent))


def _run_quiet(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a command WITHOUT showing its (noisy) output. Returns (exit code, last lines of the output)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    return r.returncode, tail


def _git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_repo(repo_dir: Path, version: str, info: dict[str, Any]) -> None:
    """Clone the DataLad dataset if needed and put it on the requested version tag."""
    if not (repo_dir / ".git").exists():
        rc, tail = 0, ""
        with Task(f"Cloning the dataset metadata from GitHub into {repo_dir} (first time only)") as t:
            rc, tail = _run_quiet(["datalad", "clone", REPO_URL, str(repo_dir)])
            if rc != 0:
                t.fail(f"datalad clone failed (exit code {rc})")
        if rc != 0:
            for ln in tail.splitlines():
                say(ln, 8)
            raise RuntimeError("datalad clone failed (internet connection?)")
    else:
        good(f"Using the existing DataLad repository: {repo_dir}")
    tag = _git(repo_dir, "describe", "--tags", "--exact-match", "HEAD")
    if tag != version:
        rc = 0
        with Task(f"Switching the repository to dataset version {version} (now: {tag or 'no tag'})") as t:
            rc, _ = _run_quiet(["git", "-C", str(repo_dir), "checkout", "--quiet", version])
            if rc != 0:
                t.fail(f"git checkout failed (exit code {rc})")
        if rc != 0:
            warn(f"Could not check out tag {version}; continuing with the current state of the repository.")
    info["repo_commit"] = _git(repo_dir, "rev-parse", "HEAD")
    info["repo_tag"] = _git(repo_dir, "describe", "--tags", "--exact-match", "HEAD")


def _copy_atomic(src: Path, dst: Path) -> None:
    """Copy and DEREFERENCE symlinks (like rsync -L). Never leaves a half-written file behind."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _https_get(url: str, dst: Path, retries: int = 3) -> tuple[str, list[str]]:
    """Returns (status, messages). status = 'ok' | 'absent' (404/403: not in the dataset) | 'failed'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    messages: list[str] = []
    for attempt in range(1, retries + 1):
        err: Exception | None = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "qc_eeg_subject/1.0"})
            done = 0
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
            if total and done != total:
                raise OSError(f"incomplete download ({done} of {total} bytes)")
            os.replace(tmp, dst)
            return "ok", messages
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):      # the public S3 bucket answers 403 for keys that do not exist
                tmp.unlink(missing_ok=True)
                return "absent", messages
            err = e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            err = e
        messages.append(f"attempt {attempt}/{retries} failed for {dst.name}: {err}")
        time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    return "failed", messages


def fetch_data(sub: str, sessions: list[str], data_dir: Path, repo_dir: Path,
               method: str, version: str, force: bool) -> dict[str, Any]:
    """Make sure the EDF (+ json / events.tsv) files of the subject are in data_dir.

    Files that are already there are not downloaded again (unless force=True).
    Files that do not exist in the dataset are only reported; they never stop the script.
    """
    info: dict[str, Any] = {"method": None, "repo_commit": None, "repo_tag": None,
                            "downloaded": [], "already_present": [], "not_in_dataset": [], "failed": []}
    min_bytes = CONFIG["edf_min_size_mb"] * 1e6
    wanted = wanted_files(sub, sessions)

    def have(kind: str, rel: str) -> bool:
        p = data_dir / rel
        return p.exists() and (kind != "edf" or p.stat().st_size >= min_bytes)

    todo = [(k, rel) for k, rel in wanted if force or not have(k, rel)]
    info["already_present"] = [rel for k, rel in wanted if (k, rel) not in todo]
    edf_todo = [rel for k, rel in todo if k == "edf"]
    side_todo = [rel for k, rel in todo if k != "edf"]
    n_edf_total = sum(1 for k, _ in wanted if k == "edf")
    if not todo:
        good(f"All {n_edf_total} EDF files are already in the data folder - nothing to download.")
        info["method"] = "none (files already present)"
        return info
    if len(edf_todo) < n_edf_total:
        good(f"{n_edf_total - len(edf_todo)} of {n_edf_total} EDF files are already in the data folder.")

    if method == "auto":
        method = "datalad" if (shutil.which("datalad") and shutil.which("git-annex")) else "https"
    info["method"] = method
    say(f"Source: OpenNeuro {DATASET_ID} v{version}  |  method: {method}  |  {len(edf_todo)} EDF file(s) to download", 4)
    n = len(edf_todo)

    if method == "datalad":
        if not (shutil.which("datalad") and shutil.which("git-annex")):
            raise RuntimeError("DataLad / git-annex not found. Install them or use --download-method https.")
        _prepare_repo(repo_dir, version, info)
        absent = [rel for rel in edf_todo if not os.path.lexists(repo_dir / rel)]
        info["not_in_dataset"].extend(absent)
        _warn_absent(absent)
        for j, rel in enumerate(edf_todo, 1):
            name, src, dst = Path(rel).name, repo_dir / rel, data_dir / rel
            if rel in absent:
                continue
            rc, tail = 0, ""
            with Task(f"Downloading [{j}/{n}] {name}") as t:
                if not src.exists():                       # annex content not here yet
                    rc, tail = _run_quiet(["datalad", "get", rel], cwd=repo_dir)
                if src.exists():
                    _copy_atomic(src, dst)
                    info["downloaded"].append(rel)
                    t.detail = f"{dst.stat().st_size / 1e6:.1f} MB"
                else:
                    info["failed"].append(rel)
                    t.fail(f"content not retrieved (datalad get exit code {rc})")
            if rel in info["failed"]:
                for ln in tail.splitlines():
                    say(ln, 8)
        copied = 0
        for rel in side_todo:
            src = repo_dir / rel
            if src.exists():
                _copy_atomic(src, data_dir / rel)
                info["downloaded"].append(rel)
                copied += 1
            else:
                info["not_in_dataset"].append(rel)
        if copied:
            good(f"Copied {copied} small side file(s) (json / events.tsv)")

    elif method == "https":
        warn(f"HTTPS downloads the newest OpenNeuro snapshot, which is not guaranteed to be v{version}. "
             "Use DataLad for an exactly pinned version.")
        for j, rel in enumerate(edf_todo, 1):
            name, dst = Path(rel).name, data_dir / rel
            status, msgs = "failed", []
            with Task(f"Downloading [{j}/{n}] {name}") as t:
                status, msgs = _https_get(f"{HTTPS_BASE}/{rel}", dst)
                if status == "ok":
                    info["downloaded"].append(rel)
                    t.detail = f"{dst.stat().st_size / 1e6:.1f} MB"
                elif status == "absent":
                    info["not_in_dataset"].append(rel)
                    t.fail("not in the dataset (will be reported as MISSING)")
                else:
                    info["failed"].append(rel)
                    t.fail("download failed")
            for m in msgs:
                warn(m, 8)
            if status == "failed" and not info["downloaded"]:     # nothing worked yet: probably no connection
                raise RuntimeError("HTTPS download failed for the first file (internet connection?)")
        copied = 0
        for rel in side_todo:
            status, _ = _https_get(f"{HTTPS_BASE}/{rel}", data_dir / rel, retries=1)
            if status == "ok":
                info["downloaded"].append(rel)
                copied += 1
            elif status == "absent":
                info["not_in_dataset"].append(rel)
        if copied:
            good(f"Downloaded {copied} small side file(s) (json / events.tsv)")
    else:
        raise ValueError(f"Unknown download method: {method}")
    return info


def read_dataset_description(repo_dir: Path) -> dict[str, Any]:
    p = repo_dir / "dataset_description.json"
    if p.exists():
        try:
            dd = json.loads(p.read_text(encoding="utf-8"))
            return {k: dd[k] for k in ("Name", "BIDSVersion", "DatasetType", "License", "DatasetDOI") if k in dd}
        except Exception:
            return {}
    return {}


# ======================================================================
# QC CODE (the logic is the same as in the pilot notebook, cells 1-6)
# ======================================================================
VERDICT_RANK = {"PASS": 0, "FLAG": 1, "FAIL": 2}


def worst_verdict(verdicts, empty: str = "PASS") -> str:
    """Worst verdict in a list (FAIL > FLAG > PASS). Other values such as N/A or MISSING are ignored."""
    valid = [v for v in verdicts if v in VERDICT_RANK]
    return max(valid, key=VERDICT_RANK.__getitem__) if valid else empty


def make_issue(verdict: str, metric: str, value: Any, threshold: Any, code: str | None = None, **extra: Any) -> dict[str, Any]:
    """One finding of a layer (used later in the summary report). `code` is the short machine-readable reason,
    for example HIGH_LINE_NOISE or USABLE_NEEDS_INTERPOLATION. verdict can be FAIL, FLAG or PASS (PASS = a note)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        value = round(value, 3)
    return {"verdict": verdict, "code": code or metric.upper(), "metric": metric, "value": value,
            "threshold": threshold, **extra}


def make_result(layer: str, metrics: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Standard return value of every layer function: layer name, verdict, metrics, findings."""
    return {"layer": layer, "verdict": worst_verdict((i["verdict"] for i in issues), empty="PASS"),
            "metrics": metrics, "issues": issues}


def sort_key(key: tuple[str, str, str, str]) -> tuple:
    """Sort order for (subject, session, task, acq)."""
    sub, ses, task, acq = key
    return (sub, ses, TASKS.index(task) if task in TASKS else 99, ACQS.index(acq) if acq in ACQS else 99)


def build_inventory(data_dir: Path, sub: str, sessions: list[str]) -> pd.DataFrame:
    """Which of the expected recordings exist on disk? (expected = the sessions the subject completed)"""
    rows = []
    for ses in sessions:
        for task in TASKS:
            for acq in ACQS:
                p = recording_paths(data_dir, sub, ses, task, acq)
                rows.append({
                    "subject": sub, "session": ses, "task": task, "acq": acq,
                    "edf_present": edf_is_real(p["edf"]),
                    "edf_MB": round(p["edf"].stat().st_size / 1e6, 1) if p["edf"].exists() else 0.0,
                    "json_present": p["json"].exists(),
                    "events_tsv_present": p["events"].exists(),
                })
    return pd.DataFrame(rows)


@dataclass
class Recording:
    """One recording in memory. Signals are in microvolts (uV)."""
    sub: str
    ses: str
    task: str
    acq: str
    ch_names: list[str]
    ch_types: list[str]
    sfreq: float
    data_uv: np.ndarray                 # EEG channels only, shape (n_channels, n_samples). RAW signal
    annotations: list[dict[str, Any]]   # onset in s from the first sample, duration, description
    extra_channels: list[tuple[str, str]] | None = None   # non-EEG channels as (name, MNE type), e.g. ("Status", "stim")
    stim_data: dict[str, np.ndarray] | None = None        # raw numbers of stim channels (trigger codes, NOT in uV)
    _dt: np.ndarray | None = field(default=None, repr=False)   # cache of the baseline-corrected copy

    @property
    def data_dt(self) -> np.ndarray:
        """Baseline-corrected copy of the signal (median removed + 0.5 Hz high-pass), computed once, in RAM only."""
        if self._dt is None:
            self._dt = detrend_signal(self.data_uv, self.sfreq)
        return self._dt

    @property
    def n_times(self) -> int:
        return int(self.data_uv.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_times / self.sfreq

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.ses, self.task, self.acq)


def load_raw(data_dir: Path, sub: str, ses: str, task: str, acq: str):
    """Read one EDF recording into memory. No filtering, no re-reference, nothing is written."""
    edf = recording_paths(data_dir, sub, ses, task, acq)["edf"]
    if not edf_is_real(edf):
        raise FileNotFoundError(f"EDF file missing or too small (broken link?): {edf}")
    try:
        return mne.io.read_raw_edf(edf, preload=True, verbose="WARNING")
    except ImportError as e:
        raise ImportError(f"{e}. If it mentions 'edfio', run:  pip install edfio") from e


def raw_to_recording(raw, sub: str, ses: str, task: str, acq: str) -> Recording:
    """Copy what we need from the MNE object.

    Channel types come from the file header (via MNE). Only channels typed "eeg" are treated as EEG.
    Other channels (for example a trigger channel typed "stim") are kept apart, so they cannot spoil the
    signal statistics. MNE stores volts, so EEG is multiplied by 1e6 to get uV.
    """
    types = list(raw.get_channel_types())
    eeg_idx = [i for i, t in enumerate(types) if t == "eeg"] or list(range(len(types)))
    extra_idx = [i for i in range(len(types)) if i not in eeg_idx]
    stim_data = {raw.ch_names[i]: np.asarray(raw.get_data(picks=[i])[0], dtype=float)
                 for i in extra_idx if types[i] == "stim"}

    ann = raw.annotations
    # If the annotations have a time origin, their onset is counted from meas_date. Remove raw.first_time
    # so that onset = seconds from the first sample of this recording.
    offset = float(raw.first_time) if ann.orig_time is not None else 0.0
    annotations = [
        {"onset": float(o) - offset, "duration": float(d), "description": str(s)}
        for o, d, s in zip(ann.onset, ann.duration, ann.description)
    ]
    return Recording(
        sub=sub, ses=ses, task=task, acq=acq,
        ch_names=[raw.ch_names[i] for i in eeg_idx],
        ch_types=[types[i] for i in eeg_idx],
        sfreq=float(raw.info["sfreq"]),
        data_uv=raw.get_data(picks=eeg_idx) * 1e6,
        annotations=annotations,
        extra_channels=[(raw.ch_names[i], types[i]) for i in extra_idx],
        stim_data=stim_data,
    )


def load_recording(data_dir: Path, sub: str, ses: str, task: str, acq: str) -> Recording:
    """Load one recording and free the MNE object."""
    raw = load_raw(data_dir, sub, ses, task, acq)
    rec = raw_to_recording(raw, sub, ses, task, acq)
    del raw
    return rec


def window_std_uv(x: np.ndarray, fs: float, win_s: float = 1.0) -> np.ndarray:
    """Typical amplitude per channel, not inflated by slow drift or DC offset.

    Median over 1-second windows of the STD after a straight line is removed inside each window.
    This is a statistic only. The signal is not filtered and nothing is stored.
    """
    w = int(round(win_s * fs))
    n_win = x.shape[1] // w
    if w < 2 or n_win < 1:
        return x.std(axis=1)
    seg = x[:, : n_win * w].reshape(x.shape[0], n_win, w)
    seg = signal.detrend(seg, axis=-1, type="linear")
    return np.median(seg.std(axis=-1), axis=1)


def read_sidecar(path: Path) -> tuple[dict | None, str | None]:
    """Read the _eeg.json sidecar. Returns (content, error). (None, None) means: file not there."""
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:  # unreadable or invalid JSON
        return None, f"{type(e).__name__}: {e}"


NON_EEG_HINTS = tuple(p.upper() for p in CONFIG["non_eeg_name_hints"])


def _norm_name(name: str) -> str:
    """Lower-case a channel name and remove noise such as an 'EEG ' prefix or a '-Ref' suffix."""
    s = name.lower().strip()
    if s.startswith("eeg"):
        s = s[3:]
    s = "".join(ch for ch in s if ch.isalnum())
    return s[:-3] if s.endswith("ref") else s


def find_channel(ch_names: list[str], target: str) -> int | None:
    """Index of a channel, case-insensitive. Tries the exact name first, then the cleaned-up name."""
    lowered = [c.lower() for c in ch_names]
    if target.lower() in lowered:
        return lowered.index(target.lower())
    wanted = _norm_name(target)
    for i, c in enumerate(ch_names):
        if _norm_name(c) == wanted:
            return i
    return None


def longest_equal_run(x: np.ndarray) -> int:
    """Length of the longest run of identical consecutive samples."""
    change = np.flatnonzero(np.diff(x) != 0)
    edges = np.concatenate(([-1], change, [len(x) - 1]))
    return int(np.diff(edges).max())


def find_frozen(x: np.ndarray, fs: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Time segments in which ALL channels are frozen at the same time (no sample changes in any channel).

    Such a segment is a recording-level event (dropout, padding or start-up block), not a bad electrode.
    Returns (boolean mask of the frozen samples, list of (start_s, length_s)).
    """
    same = np.all(np.diff(x, axis=1) == 0, axis=0)
    pad = np.concatenate(([False], same, [False]))
    edges = np.flatnonzero(pad[1:] != pad[:-1])
    mask = np.zeros(x.shape[1], dtype=bool)
    segments = []
    for s0, e0 in zip(edges[::2], edges[1::2]):
        length = e0 - s0 + 1
        if length >= CONFIG["frozen_min_samples"]:
            mask[s0:s0 + length] = True
            segments.append((float(s0 / fs), float(length / fs)))
    return mask, segments


# ---------------------------------------------------------------- 1) HEADER AND METADATA
def header_check(rec: Recording, sidecar: dict | None = None, sidecar_error: str | None = None) -> dict[str, Any]:
    """Layer 1: EEG channel count, sampling rate, duration, channel names, O1/O2/Oz, JSON sidecar.

    Only channels typed "eeg" in the file header are counted. Auxiliary channels (trigger, EOG, ECG ...)
    are listed but never counted as EEG.
    """
    c = CONFIG
    issues: list[dict[str, Any]] = []
    names, dur = rec.ch_names, rec.duration_s
    aux = rec.extra_channels or []
    n_empty = sum(1 for n in names if not n.strip())
    duplicates = sorted(n for n, k in Counter(names).items() if k > 1)
    non_eeg_like = [n for n in names + [a[0] for a in aux] if any(h in n.upper() for h in NON_EEG_HINTS)]
    required = {}
    for target in c["required_channels"]:
        idx = find_channel(names, target)
        required[target] = names[idx] if idx is not None else None
    missing = [t for t, v in required.items() if v is None]

    if len(names) != c["expected_n_eeg_channels"]:
        issues.append(make_issue("FAIL", "n_eeg_channels", len(names), f"== {c['expected_n_eeg_channels']}"))
    if abs(rec.sfreq - c["expected_sfreq_hz"]) > 1e-6:
        issues.append(make_issue("FAIL", "sfreq_hz", rec.sfreq, f"== {c['expected_sfreq_hz']:.0f}"))
    lo, hi = c["duration_ok_range_s"]
    if dur < c["duration_fail_below_s"]:
        issues.append(make_issue("FAIL", "duration_s", dur, f">= {c['duration_fail_below_s']:.0f} s"))
    elif not (lo <= dur <= hi):
        issues.append(make_issue("FLAG", "duration_s", dur, f"between {lo:.0f} and {hi:.0f} s"))
    if n_empty or duplicates:
        issues.append(make_issue("FAIL", "channel_names_unique_and_not_empty",
                                 f"{n_empty} empty, duplicates: {duplicates}", "all names unique and not empty"))
    if missing:
        issues.append(make_issue("FAIL", "required_channels_missing", ", ".join(missing), "O1, O2 and Oz must exist"))

    json_sf = json_n = None
    if sidecar is not None:
        json_sf, json_n = sidecar.get("SamplingFrequency"), sidecar.get("EEGChannelCount")
        try:
            if json_sf is not None and abs(float(json_sf) - rec.sfreq) > 1e-6:
                issues.append(make_issue("FLAG", "sidecar_SamplingFrequency", json_sf, f"equal to the file value {rec.sfreq}"))
            if json_n is not None and int(json_n) != len(names):
                issues.append(make_issue("FLAG", "sidecar_EEGChannelCount", json_n, f"equal to the file value {len(names)}"))
        except (TypeError, ValueError):
            issues.append(make_issue("FLAG", "sidecar_values_not_numeric", f"{json_sf}, {json_n}", "numbers"))
    elif sidecar_error:
        issues.append(make_issue("FLAG", "sidecar_unreadable", sidecar_error, "valid JSON"))

    metrics = {
        "n_eeg_channels": len(names), "aux_channels": [f"{n} ({t})" for n, t in aux],
        "sfreq_hz": rec.sfreq, "duration_s": round(dur, 2), "n_samples": rec.n_times,
        "non_eeg_like_names": non_eeg_like, "required_channels_found": required,
        "sidecar_found": sidecar is not None, "sidecar_SamplingFrequency": json_sf,
        "sidecar_EEGChannelCount": json_n, "channel_names": names,
    }
    return make_result("header", metrics, issues)


# ---------------------------------------------------------------- 2) CHANNEL INTEGRITY and 3) AMPLITUDE
def detrend_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """Baseline-corrected COPY of the signal, in RAM only (the raw data stay untouched).

    The channel median is removed first (kills the DC offset), then slow drift is removed with a zero-phase
    Butterworth high-pass (default 0.5 Hz) or a straight line (CONFIG["detrend_method"]).
    """
    method = CONFIG["detrend_method"]
    if method == "none":
        return x
    xm = x - np.median(x, axis=1, keepdims=True)
    if method == "linear":
        return signal.detrend(xm, axis=1, type="linear")
    if method != "highpass":
        raise ValueError(f"Unknown detrend_method: {method!r}")
    sos = signal.butter(CONFIG["detrend_hp_order"], CONFIG["detrend_hp_hz"], btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, xm, axis=1)


def spatial_correlation(rec: Recording) -> dict[str, Any]:
    """|Pearson r| of every channel with the MEDIAN of all channels, in 5-s windows (band 1-45 Hz).

    A healthy EEG channel shares a part of its signal with the other channels (volume conduction, reference).
    A noisy or disconnected channel that is neither flat nor clipped does not. The median over channels is used
    as the reference because one bad channel cannot move a median.
    Returns per channel: median over windows ("median"), share of windows below the threshold ("low_frac"),
    and the full channel x window matrix ("windows", used for the figure).
    """
    c = CONFIG
    fs = rec.sfreq
    q = max(1, int(round(fs / c["corr_target_fs_hz"])))
    lo, hi = c["corr_band_hz"]
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, rec.data_dt, axis=1)[:, ::q]       # band-passed, then down-sampled (hi < new Nyquist)
    fs_y = fs / q
    n_ch, n = y.shape
    w = int(round(c["corr_window_s"] * fs_y))
    n_win = n // w
    if n_win < 1:
        w, n_win = n, 1
    seg = y[:, : n_win * w].reshape(n_ch, n_win, w)
    ref = np.median(seg, axis=0)                                    # (n_win, w): median over channels
    sc = seg - seg.mean(axis=-1, keepdims=True)
    rc = ref - ref.mean(axis=-1, keepdims=True)
    num = (sc * rc[None]).sum(axis=-1)
    den = np.sqrt((sc ** 2).sum(axis=-1)) * np.sqrt((rc ** 2).sum(axis=-1))[None]
    r = np.abs(np.divide(num, den, out=np.zeros_like(num), where=den > 0))
    return {"median": np.median(r, axis=1), "low_frac": (r < c["corr_threshold"]).mean(axis=1),
            "windows": r.astype(np.float32), "window_s": w / fs_y}


def compute_channel_table(rec: Recording, corr: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-channel statistics in uV. One row per EEG channel.

    RAW signal      : flat, clipping, frozen samples, DC offset, drift (digital / baseline properties).
    BASELINE-CORRECTED copy (rec.data_dt): amplitude, variance, skewness, kurtosis, percentiles, spatial correlation.
    """
    c = CONFIG
    x = rec.data_uv
    xd = rec.data_dt
    n_ch, n = x.shape
    p_lo, p_hi = np.percentile(xd, [0.5, 99.5], axis=1)
    med = np.median(x, axis=1)

    # Clipping is judged without the segments in which all channels are frozen together (reported separately).
    frozen, _ = find_frozen(x, rec.sfreq)
    xc = x[:, ~frozen] if (frozen.any() and (~frozen).sum() > 2) else x
    cmin, cmax = xc.min(axis=1), xc.max(axis=1)
    n_hit = (xc == cmax[:, None]).sum(axis=1) + (xc == cmin[:, None]).sum(axis=1)
    clip_frac = np.where(cmax > cmin, n_hit / xc.shape[1], 0.0)

    corr = spatial_correlation(rec) if corr is None else corr
    w = max(1, int(min(60.0, n / rec.sfreq / 2.0) * rec.sfreq))
    df = pd.DataFrame({
        "channel": rec.ch_names,
        "std_uv": xd.std(axis=1),
        "std_clean_uv": xc.std(axis=1),                          # raw STD without the frozen segments (used for "flat")
        "window_std_uv": window_std_uv(xd, rec.sfreq),           # typical 1-s amplitude of the baseline-corrected signal
        "variance_uv2": xd.var(axis=1),
        "skewness": stats.skew(xd, axis=1),
        "kurtosis": stats.kurtosis(xd, axis=1),                  # excess kurtosis (normal distribution = 0)
        "median_uv": med,                                        # raw: DC offset
        "p_low_uv": p_lo, "p_high_uv": p_hi, "range_p_uv": p_hi - p_lo,
        "ptp_uv": xd.max(axis=1) - xd.min(axis=1),
        "clip_frac": clip_frac,
        "longest_equal_run_s": [longest_equal_run(xc[i]) / rec.sfreq for i in range(n_ch)],
        "drift_uv": np.median(x[:, -w:], axis=1) - np.median(x[:, :w], axis=1),   # raw: first vs last minute
        "corr_median": corr["median"],
        "corr_low_frac": corr["low_frac"],
    })
    med_win = float(df["window_std_uv"].median())
    df["flat"] = df["std_clean_uv"] < c["flat_std_uv"]
    df["clipped"] = (~df["flat"]) & (df["clip_frac"] > c["clip_fraction"])
    df["noisy"] = ((~df["flat"]) & (~df["clipped"])
                   & (df["window_std_uv"] > np.maximum(c["noisy_ratio"] * med_win, c["noisy_min_uv"])))
    # spatial correlation: channels that are not flat/clipped but unrelated to all the others
    df["corr_below_thr"] = (~df["flat"]) & (~df["clipped"]) & (df["corr_median"] < c["corr_threshold"])
    unreliable = bool(df["corr_below_thr"].mean() > c["corr_unreliable_fraction"])
    df["corr_ref_unreliable"] = unreliable                       # most channels below r: the reference itself is doubtful
    df["uncorrelated"] = df["corr_below_thr"] & (not unreliable)
    df["bad"] = df["flat"] | df["clipped"] | df["noisy"] | df["uncorrelated"]
    crit = {_norm_name(k) for k in c["critical_channels"]}
    df["is_critical"] = [_norm_name(nm) in crit for nm in rec.ch_names]
    df["dc_offset_info"] = df["median_uv"].abs() > c["dc_offset_info_uv"]
    df["drift_info"] = df["drift_uv"].abs() > c["drift_info_uv"]
    return df


def integrity_check(rec: Recording, ch_table: pd.DataFrame | None = None) -> dict[str, Any]:
    """Layer 2: flat, clipped, noisy and uncorrelated channels, frozen segments, and what can be repaired.

    Decision logic (first-pass screening: keep what can be saved, reject only what cannot):
      * 2 or more CRITICAL channels bad (Cz, Fz, Pz, O1, O2; e.g. both O1 and O2)  -> FAIL
      * more than 6 bad channels in total (too many to interpolate)                -> FAIL
      * 1 critical channel bad                                                      -> FLAG CRITICAL_CHANNEL_BAD
      * 1..6 bad channels, nothing worse                                            -> PASS, label USABLE_NEEDS_INTERPOLATION
    """
    c = CONFIG
    t = compute_channel_table(rec) if ch_table is None else ch_table
    flat, clipped, noisy, uncorr = (t.loc[t[k], "channel"].tolist() for k in ("flat", "clipped", "noisy", "uncorrelated"))
    bad_rows = t[t["bad"]]
    bad = bad_rows["channel"].tolist()
    n_bad = len(bad)
    crit_bad = bad_rows.loc[bad_rows["is_critical"], "channel"].tolist()
    other_bad = [b for b in bad if b not in crit_bad]
    detail = ", ".join(
        f"{r.channel} ({'+'.join(k for k in ('flat', 'clipped', 'noisy', 'uncorrelated') if getattr(r, k))})"
        for r in bad_rows.itertuples())

    issues: list[dict[str, Any]] = []
    crit_txt = ", ".join(c["critical_channels"])
    if len(crit_bad) >= c["critical_bad_fail_min"]:
        issues.append(make_issue("FAIL", "critical_channels_bad", ", ".join(crit_bad),
                                 f"fewer than {c['critical_bad_fail_min']} of {crit_txt} bad", code="CRITICAL_CHANNELS_BAD"))
    elif len(crit_bad) == 1:
        issues.append(make_issue("FLAG", "critical_channel_bad", crit_bad[0],
                                 f"none of {crit_txt} bad (one can be interpolated)", code="CRITICAL_CHANNEL_BAD"))
    if n_bad > c["interpolatable_max_channels"]:
        issues.append(make_issue("FAIL", "bad_channels_total", f"{n_bad}: {detail}",
                                 f"at most {c['interpolatable_max_channels']} (more cannot be interpolated reliably)",
                                 code="TOO_MANY_BAD_CHANNELS"))
    interpolation_needed = n_bad > 0 and not any(i["verdict"] == "FAIL" for i in issues)
    if interpolation_needed:
        issues.append(make_issue("PASS", "bad_channels_to_interpolate", detail,
                                 f"1-{c['interpolatable_max_channels']} bad channels can be interpolated; the recording stays usable",
                                 code="USABLE_NEEDS_INTERPOLATION"))
    if bool(t["corr_ref_unreliable"].any()):
        share = float(t["corr_below_thr"].mean())
        issues.append(make_issue("FLAG", "spatial_correlation_reference_unreliable", f"{share:.0%} of channels below r={c['corr_threshold']}",
                                 f"at most {c['corr_unreliable_fraction']:.0%} of channels below the threshold "
                                 "(otherwise the correlation criterion is not applied)", code="LOW_GLOBAL_SPATIAL_CORRELATION"))

    _, segs = find_frozen(rec.data_uv, rec.sfreq)
    frozen_s = float(sum(length for _, length in segs))
    if frozen_s > c["global_frozen_fail_fraction"] * rec.duration_s:
        issues.append(make_issue("FAIL", "all_channels_frozen_s", frozen_s,
                                 f"less than {c['global_frozen_fail_fraction']:.0%} of the recording", code="RECORDING_FROZEN"))
    elif frozen_s > c["global_frozen_flag_s"]:
        issues.append(make_issue("FLAG", "all_channels_frozen_s", frozen_s, f"at most {c['global_frozen_flag_s']} s",
                                 code="FROZEN_SEGMENT"))

    metrics = {
        "n_bad_channels": n_bad, "bad_channels": bad, "bad_detail": detail,
        "n_flat": len(flat), "n_clipped": len(clipped), "n_noisy": len(noisy), "n_uncorrelated": len(uncorr),
        "flat_channels": flat, "clipped_channels": clipped, "noisy_channels": noisy, "uncorrelated_channels": uncorr,
        "n_critical_bad": len(crit_bad), "critical_bad_channels": crit_bad, "n_noncritical_bad": len(other_bad),
        "interpolation_needed": bool(interpolation_needed),
        "median_spatial_corr": float(t["corr_median"].median()),
        "corr_reference_reliable": not bool(t["corr_ref_unreliable"].any()),
        "max_clip_fraction": float(t["clip_frac"].max()),
        "frozen_s": round(frozen_s, 3),                                   # all channels frozen together
        "frozen_segments": "; ".join(f"{a:.1f}s+{b:.1f}s" for a, b in segs[:5]),
    }
    return make_result("integrity", metrics, issues)


def amplitude_check(rec: Recording, ch_table: pd.DataFrame | None = None) -> dict[str, Any]:
    """Layer 3: is the amplitude in a physiological range? (visual trace inspection, as numbers)

    Uses the typical 1-second STD of every channel on the BASELINE-CORRECTED signal (slow drift cannot inflate it).
    FAIL if the median over channels is outside the expected range (probable unit/scale problem).
    FLAG if more than 10% of channels are above it. DC offset and drift are reported but never change the verdict.
    Variance, skewness and kurtosis (baseline-corrected) are reported as numbers.
    """
    c = CONFIG
    t = compute_channel_table(rec) if ch_table is None else ch_table
    typical = float(t["window_std_uv"].median())
    frac_high = float((t["window_std_uv"] > c["typical_amp_max_uv"]).mean())
    issues = []
    if not (c["typical_amp_min_uv"] <= typical <= c["typical_amp_max_uv"]):
        issues.append(make_issue("FAIL", "typical_amplitude_uv", typical,
                                 f"between {c['typical_amp_min_uv']:.0f} and {c['typical_amp_max_uv']:.0f} uV (probable unit/scale problem)",
                                 code="AMPLITUDE_OUT_OF_RANGE"))
    elif frac_high > c["high_amp_channel_fraction"]:
        issues.append(make_issue("FLAG", "fraction_channels_high_amplitude", frac_high,
                                 f"at most {c['high_amp_channel_fraction']:.0%} of channels above {c['typical_amp_max_uv']:.0f} uV",
                                 code="HIGH_AMPLITUDE_CHANNELS"))
    metrics = {
        "typical_amplitude_uv": typical,                                   # median over channels of the 1-s STD
        "whole_recording_std_uv": float(t["std_uv"].median()),            # baseline-corrected
        "median_variance_uv2": float(t["variance_uv2"].median()),
        "median_skewness": float(t["skewness"].median()),
        "median_kurtosis": float(t["kurtosis"].median()),
        "median_range_0.5_99.5_uv": float(t["range_p_uv"].median()),
        "fraction_high_amplitude": frac_high,
        "n_dc_offset_info": int(t["dc_offset_info"].sum()),               # raw signal
        "n_drift_info": int(t["drift_info"].sum()),                       # raw signal
    }
    return make_result("amplitude", metrics, issues)


# ---------------------------------------------------------------- 4) SPECTRAL VALIDATION (Welch PSD)
def compute_psd(rec: Recording) -> dict[str, Any]:
    """Welch PSD (uV^2/Hz) of ALL EEG channels, on the in-memory signal only (nothing is filtered or saved).

    Settings (from CONFIG): Hann window, 4 s long, 50% overlap, constant detrend per segment.
    """
    c = CONFIG
    nperseg = min(int(round(c["welch_window_s"] * rec.sfreq)), rec.n_times)
    freqs, psd = signal.welch(rec.data_uv, fs=rec.sfreq, window="hann", nperseg=nperseg,
                              noverlap=int(nperseg * c["welch_overlap"]), detrend="constant", axis=-1)
    return {"freqs": freqs, "psd": psd.astype(np.float32), "channels": list(rec.ch_names), "sfreq": rec.sfreq}


def _band(f: np.ndarray, band) -> np.ndarray:
    return (f >= band[0]) & (f <= band[1])


def _flank_mean(f: np.ndarray, p: np.ndarray, flanks) -> float:
    m = np.zeros(f.shape, dtype=bool)
    for b in flanks:
        m |= _band(f, b)
    return float(p[m].mean()) if m.any() else float("nan")


def alpha_metrics(f: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Alpha peak frequency, prominence (dB over the flanking bands) and band power (uV^2) of one PSD."""
    m = _band(f, CONFIG["alpha_band_hz"])
    if not m.any():
        return {"peak_hz": np.nan, "prominence_db": np.nan, "power_uv2": np.nan}
    i = int(np.argmax(p[m]))
    peak, flank = float(p[m][i]), _flank_mean(f, p, CONFIG["alpha_flank_hz"])
    prom = 10 * np.log10(peak / flank) if (flank > 0 and peak > 0) else np.nan
    return {"peak_hz": float(f[m][i]), "prominence_db": float(prom), "power_uv2": float(trapezoid(p[m], f[m]))}


def line_db(f: np.ndarray, p: np.ndarray, center: float) -> float:
    """Line noise in dB: highest PSD value near `center` versus the mean PSD of the two flanking bands."""
    lo, hi = CONFIG["line_flank_offsets_hz"]
    m = _band(f, (center - CONFIG["line_peak_halfwidth_hz"], center + CONFIG["line_peak_halfwidth_hz"]))
    flank = _flank_mean(f, p, ((center - hi, center - lo), (center + lo, center + hi)))
    if not m.any() or not (flank > 0):
        return float("nan")
    return float(10 * np.log10(p[m].max() / flank))


def channel_spectral_table(psd: dict[str, Any]) -> pd.DataFrame:
    """Alpha power/prominence and 50 Hz line noise for EVERY channel (used for the scalp maps)."""
    f, rows = psd["freqs"], []
    for ch, p in zip(psd["channels"], psd["psd"]):
        p = p.astype(float)
        a = alpha_metrics(f, p)
        rows.append({"alpha_power_uv2": a["power_uv2"], "alpha_prominence_db": a["prominence_db"],
                     "line50_db": line_db(f, p, CONFIG["line_hz"])})
    return pd.DataFrame(rows)


def occipital_metrics(psd: dict[str, Any]) -> dict[str, Any]:
    """Alpha, line-noise and broadband-power numbers on O1/O2/Oz (per channel and combined)."""
    idx = {t: find_channel(psd["channels"], t) for t in CONFIG["psd_channels"]}
    idx = {t: i for t, i in idx.items() if i is not None}
    f, out = psd["freqs"], {"channels": list(idx)}
    band = _band(f, CONFIG["dead_spectrum_band_hz"])
    peaks, proms, powers, l1, l2, l3, broad = [], [], [], [], [], [], []
    for t, i in idx.items():
        p = psd["psd"][i].astype(float)
        a = alpha_metrics(f, p)
        out[f"alpha_power_uv2_{t}"] = a["power_uv2"]
        peaks.append(a["peak_hz"]); proms.append(a["prominence_db"]); powers.append(a["power_uv2"])
        l1.append(line_db(f, p, CONFIG["line_hz"]))
        l2.append(line_db(f, p, CONFIG["harmonics_hz"][0]))
        l3.append(line_db(f, p, CONFIG["harmonics_hz"][1]))
        broad.append(float(trapezoid(p[band], f[band])) if band.any() else float("nan"))
    nm = lambda v, fn: float(fn(v)) if v else np.nan          # noqa: E731
    out.update(alpha_peak_hz=nm(peaks, np.nanmean), alpha_prominence_db=nm(proms, np.nanmedian),
               alpha_power_uv2=nm(powers, np.nanmean), line50_db=nm(l1, np.nanmax),
               line100_db=nm(l2, np.nanmax), line150_db=nm(l3, np.nanmax),
               broadband_power_uv2=nm(broad, np.nanmedian),
               dead_spectrum=bool(len(broad) > 0 and all(np.nan_to_num(b, nan=0.0) < CONFIG["dead_spectrum_power_uv2"] for b in broad)))
    return out


def spectral_layer(psd_ec: dict | None, psd_eo: dict | None,
                   clipped_frac: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    """Layer 4: spectral validation for ONE pair (same subject, session and acq).

    Returns {"EyesClosed": result, "EyesOpen": result}.  `clipped_frac` = share of clipped channels per task.
    - Dead spectrum (no signal power at all on O1, O2 and Oz): the only alpha-related FAIL.
    - Alpha blocking (power Eyes Closed / Eyes Open on O1, O2, Oz) and alpha peak prominence are FLAGS only:
      10-15% of healthy people are "alpha non-responders". Codes: WEAK_ALPHA_BLOCKING, NO_ALPHA_BLOCKING,
      LOW_ALPHA_PROMINENCE (the metric ALPHA_NON_RESPONDER_POSSIBLE is reported when both are weak).
    - Line noise (50 Hz above its flanks): above 20 dB = FLAG LINE_NOISE, above 30 dB = FLAG HIGH_LINE_NOISE
      (a notch filter removes it later). FAIL only when the high line noise comes with clipping of at least half
      of the channels (permanent saturation), code LINE_NOISE_SATURATION.
    """
    c = CONFIG
    clipped_frac = clipped_frac or {}
    occ = {"EyesClosed": occipital_metrics(psd_ec) if psd_ec else None,
           "EyesOpen": occipital_metrics(psd_eo) if psd_eo else None}
    ratios: dict[str, float] = {}
    if occ["EyesClosed"] and occ["EyesOpen"]:
        for t in occ["EyesClosed"]["channels"]:
            if t in occ["EyesOpen"]["channels"]:
                a_eo = occ["EyesOpen"][f"alpha_power_uv2_{t}"]
                ratios[t] = float(occ["EyesClosed"][f"alpha_power_uv2_{t}"] / a_eo) if a_eo > 0 else np.nan
    valid = [v for v in ratios.values() if not np.isnan(v)]
    ratio_med = float(np.median(valid)) if valid else np.nan

    out: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        o = occ[task]
        if o is None:
            continue
        issues: list[dict[str, Any]] = []
        if not o["channels"]:
            issues.append(make_issue("FAIL", "occipital_channels_missing", "none of O1/O2/Oz found", "O1, O2, Oz needed",
                                     code="OCCIPITAL_CHANNELS_MISSING"))
            out[task] = make_result("spectral", {"n_psd_channels": 0}, issues)
            continue
        metrics = {k: v for k, v in o.items()
                   if k != "channels" and not k.startswith("alpha_power_uv2_")}
        metrics["notch_suggested"] = bool(o["line50_db"] > c["notch_suggested_db"])

        # --- the only spectral FAIL besides saturation: no signal power at all ---
        if o["dead_spectrum"]:
            issues.append(make_issue("FAIL", "occipital_broadband_power_uv2", o["broadband_power_uv2"],
                                     f"at least {c['dead_spectrum_power_uv2']} uV^2 in {c['dead_spectrum_band_hz'][0]:.0f}-"
                                     f"{c['dead_spectrum_band_hz'][1]:.0f} Hz on O1/O2/Oz (spectrum is flat / dead)",
                                     code="DEAD_SPECTRUM"))

        # --- line noise: a notch filter fixes it, so it is a FLAG unless the sensors are saturated ---
        db, sat = o["line50_db"], float(clipped_frac.get(task, 0.0))
        if db > c["line_high_db"]:
            if sat >= c["line_saturation_channel_fraction"]:
                issues.append(make_issue("FAIL", "line_50Hz_with_saturation", f"{db:.1f} dB, {sat:.0%} of channels clipped",
                                         f"at most {c['line_high_db']:.0f} dB, or fewer than "
                                         f"{c['line_saturation_channel_fraction']:.0%} of channels clipped", code="LINE_NOISE_SATURATION"))
            else:
                issues.append(make_issue("FLAG", "line_50Hz_db", db, f"at most {c['line_high_db']:.0f} dB (notch filter needed)",
                                         code="HIGH_LINE_NOISE"))
        elif db > c["line_flag_db"]:
            issues.append(make_issue("FLAG", "line_50Hz_db", db, f"at most {c['line_flag_db']:.0f} dB", code="LINE_NOISE"))

        # --- alpha: FLAG only (alpha non-responders are healthy) ---
        weak_peak = (task == "EyesClosed" and o["alpha_prominence_db"] < c["alpha_prominence_min_db"])
        if weak_peak:
            issues.append(make_issue("FLAG", "alpha_peak_prominence_db", o["alpha_prominence_db"],
                                     f"at least {c['alpha_prominence_min_db']:.0f} dB in Eyes Closed", code="LOW_ALPHA_PROMINENCE"))
        if valid:
            metrics["alpha_ratio_EC_EO"] = ratio_med
            if ratio_med < c["alpha_ratio_pass"]:
                if max(valid) < c["alpha_no_blocking_ratio"]:
                    issues.append(make_issue("FLAG", "alpha_ratio_EC_EO", ratio_med,
                                             f"at least {c['alpha_no_blocking_ratio']} on one channel", code="NO_ALPHA_BLOCKING"))
                else:
                    issues.append(make_issue("FLAG", "alpha_ratio_EC_EO", ratio_med,
                                             f"median at least {c['alpha_ratio_pass']}", code="WEAK_ALPHA_BLOCKING"))
        else:
            metrics["alpha_ratio_EC_EO"] = np.nan
            issues.append(make_issue("FLAG", "alpha_ratio_EC_EO", "not computed", "needs both Eyes Closed and Eyes Open",
                                     code="ALPHA_RATIO_NOT_COMPUTED"))
        metrics["alpha_non_responder_possible"] = bool(task == "EyesClosed" and weak_peak and valid
                                                       and ratio_med < c["alpha_ratio_pass"])
        out[task] = make_result("spectral", metrics, issues)
    return out


# ---------------------------------------------------------------- 5) CONSISTENCY and trigger/event information
def consistency_layer(summaries: dict[tuple, dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """All recordings should have the same channel names (and order) and the same sampling rate.

    `summaries` maps (subject, session, task, acq) -> {"ch_names": tuple, "sfreq": float}.
    A recording that differs from the most common layout gets a FAIL.
    """
    if not summaries:
        return {}
    ref_names = Counter(s["ch_names"] for s in summaries.values()).most_common(1)[0][0]
    ref_sf = Counter(s["sfreq"] for s in summaries.values()).most_common(1)[0][0]
    out = {}
    for key, s in summaries.items():
        issues = []
        if s["ch_names"] != ref_names:
            issues.append(make_issue("FAIL", "channel_names_or_order_differ", "different", "same as the other recordings"))
        if s["sfreq"] != ref_sf:
            issues.append(make_issue("FAIL", "sfreq_differs", s["sfreq"], f"same as the other recordings ({ref_sf})"))
        out[key] = make_result("consistency", {"same_layout_as_cohort": not issues}, issues)
    return out


def stim_events(rec: Recording) -> dict[str, Any]:
    """Summarise trigger channels (type "stim"): number of events, values, first/last time. Codes are NOT interpreted.

    An event = the channel changes to a value that is not its most common (baseline) value.
    """
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
    """Trigger/event information (metadata only, no verdict): EDF annotations, trigger channel, events.tsv."""
    info = {"edf_annotations": len(rec.annotations), **stim_events(rec), "events_tsv_rows": np.nan}
    if events_path is not None and events_path.exists():
        try:
            info["events_tsv_rows"] = len(pd.read_csv(events_path, sep="\t"))
        except Exception as e:
            info["events_tsv_rows"] = f"unreadable ({type(e).__name__})"
    return info


OVERVIEW_STEP = 25      # the whole-recording trace figure keeps every 25th sample (40 Hz for 1000 Hz data)


def analyse_recording(data_dir: Path, rec: Recording) -> dict[str, Any]:
    """Run layers 1-3 and compute the PSD of ONE recording. Layers that need several recordings come later.

    Returns a dict with: layers, ch_table (per channel), psd (all channels), excerpt, overview, corr windows,
    summary, info.
    """
    paths = recording_paths(data_dir, rec.sub, rec.ses, rec.task, rec.acq)
    sidecar, sidecar_error = read_sidecar(paths["json"])
    corr = spatial_correlation(rec)                                    # also creates the baseline-corrected copy
    ch = compute_channel_table(rec, corr)
    psd = compute_psd(rec)
    ch = pd.concat([ch, channel_spectral_table(psd)], axis=1)          # same channel order, so concat by position
    layers = {"header": header_check(rec, sidecar, sidecar_error),
              "integrity": integrity_check(rec, ch),
              "amplitude": amplitude_check(rec, ch)}
    n, fs = rec.n_times, rec.sfreq
    a = int(max(0, n // 2 - 3 * fs))
    seg = rec.data_dt[:, a:a + int(6 * fs)]                            # 6 s from the middle, baseline-corrected
    excerpt = seg.astype(np.float32)                                   # (the median is already removed)

    # whole recording on the occipital channels (down-sampled for display) + frozen segments, for figure F
    traces = {}
    for target in CONFIG["psd_channels"]:
        i = find_channel(rec.ch_names, target)
        if i is not None:
            x = rec.data_uv[i]
            traces[target] = (x - np.median(x))[::OVERVIEW_STEP].astype(np.float32)
    _, frozen_segs = find_frozen(rec.data_uv, rec.sfreq)
    overview = {"sfreq": rec.sfreq, "step": OVERVIEW_STEP, "traces": traces, "frozen_segs": frozen_segs}

    return {"layers": layers, "ch_table": ch, "psd": psd, "excerpt": excerpt, "overview": overview,
            "corr_windows": corr["windows"], "corr_window_s": corr["window_s"],
            "summary": {"ch_names": tuple(rec.ch_names), "sfreq": rec.sfreq, "clipped_frac": float(ch["clipped"].mean())},
            "info": events_info(rec, paths["events"])}


# ======================================================================
# RUN ALL RECORDINGS OF THE SUBJECT (one at a time, to keep memory low)
# ======================================================================
@dataclass
class QCState:
    records: dict[tuple, dict] = field(default_factory=dict)        # key -> {"status","error","layers","info"}
    psd: dict[tuple, dict] = field(default_factory=dict)            # PSD of all channels (small arrays)
    ch_tables: dict[tuple, pd.DataFrame] = field(default_factory=dict)
    excerpts: dict[tuple, np.ndarray] = field(default_factory=dict)  # 6 s of signal per recording
    overviews: dict[tuple, dict] = field(default_factory=dict)
    summaries: dict[tuple, dict] = field(default_factory=dict)
    corr_windows: dict[tuple, np.ndarray] = field(default_factory=dict)   # |r| per channel and 5-s window
    corr_window_s: dict[tuple, float] = field(default_factory=dict)


def run_qc(data_dir: Path, sub: str, sessions: list[str], debug: bool = False) -> tuple[QCState, pd.DataFrame]:
    st = QCState()
    inventory = build_inventory(data_dir, sub, sessions)
    n_tot = len(inventory)
    n_ok = int(inventory["edf_present"].sum())
    if n_ok == n_tot:
        good(f"{n_ok} of {n_tot} expected EDF files are on disk.")
    else:
        warn(f"{n_ok} of {n_tot} expected EDF files are on disk. The missing ones are reported as MISSING; "
             "the others are analysed normally.")

    for i, r in enumerate(inventory.itertuples(index=False), 1):
        key = (r.subject, r.session, r.task, r.acq)
        label = f"recording {i}/{n_tot}: {r.session}  {r.task}  {r.acq}"
        if not r.edf_present:
            edf = recording_paths(data_dir, *key)["edf"]       # reported below as MISSING; the others go on normally
            st.records[key] = {"status": "MISSING", "error": f"EDF file missing or too small: {edf}",
                               "layers": {}, "info": {}}
            continue
        rec = None
        try:
            with Task(f"Analysing {label}") as t:
                try:
                    rec = load_recording(data_dir, *key)
                    out = analyse_recording(data_dir, rec)
                    st.records[key] = {"status": "OK", "error": None, "layers": out["layers"], "info": out["info"]}
                    st.ch_tables[key], st.psd[key] = out["ch_table"], out["psd"]
                    st.excerpts[key], st.overviews[key], st.summaries[key] = out["excerpt"], out["overview"], out["summary"]
                    st.corr_windows[key], st.corr_window_s[key] = out["corr_windows"], out["corr_window_s"]
                except NameError:              # a missing function is a script problem, not a file problem
                    raise
                except Exception as e:         # an unreadable file becomes a FAIL row
                    if debug:
                        raise
                    msg = f"{type(e).__name__}: {e}"
                    st.records[key] = {"status": "ERROR", "error": msg, "layers": {}, "info": {}}
                    t.fail(f"could not be processed -> FAIL ({msg})")
        finally:
            rec = None
            gc.collect()

    # ---- layer 4: Eyes Closed vs Eyes Open (same subject, session, acq) ----
    for ses, acq in sorted({(k[1], k[3]) for k in st.records}):
        k_ec, k_eo = (sub, ses, "EyesClosed", acq), (sub, ses, "EyesOpen", acq)
        clip = {"EyesClosed": st.summaries.get(k_ec, {}).get("clipped_frac", 0.0),
                "EyesOpen": st.summaries.get(k_eo, {}).get("clipped_frac", 0.0)}
        pair = spectral_layer(st.psd.get(k_ec), st.psd.get(k_eo), clip)
        for task, k in (("EyesClosed", k_ec), ("EyesOpen", k_eo)):
            if task in pair and k in st.records and st.records[k]["status"] == "OK":
                st.records[k]["layers"]["spectral"] = pair[task]

    # ---- layer 5: same channel layout and sampling rate in all recordings of the subject ----
    for key, res in consistency_layer(st.summaries).items():
        st.records[key]["layers"]["consistency"] = res
    return st, inventory


# ======================================================================
# RESULT TABLES (pilot notebook, cell 7 and 8)
# ======================================================================
LAYER_ORDER = ["header", "integrity", "amplitude", "spectral", "consistency"]
LAYER_PREFIX = {"header": "hdr", "integrity": "int", "amplitude": "amp", "spectral": "spec", "consistency": "con"}
SKIP_KEYS = {"channel_names", "required_channels_found"}
OUTPUT_FILES = ["df_recordings.csv", "df_recordings_full.csv", "df_subjects.csv", "df_channels.csv",
                "qc_pivot.csv", "qc_issues.csv", "qc_signoff.csv", "qc_report.txt", "run_manifest.json", "run_log.txt"]


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Turn a metrics dict into table columns (numbers and text; lists become text)."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(map(str, v))
        elif isinstance(v, (float, np.floating)):
            v = round(float(v), 3)
        elif isinstance(v, np.generic):
            v = v.item()
        out[f"{prefix}_{k}"] = v
    return out


def metric(layers: dict, layer: str, name: str, default: Any = np.nan) -> Any:
    """One metric of one layer (or a default if the layer or metric is missing)."""
    return layers.get(layer, {}).get("metrics", {}).get(name, default)


LABEL_RANK = {"USABLE": 0, "USABLE_WITH_FLAGS": 1, "USABLE_NEEDS_INTERPOLATION": 2, "REJECT": 3}
LABEL_TEXT = {
    "USABLE": "usable as it is",
    "USABLE_WITH_FLAGS": "usable; flags (for example line noise, weak alpha) are worth a look",
    "USABLE_NEEDS_INTERPOLATION": "usable after interpolating the bad channels",
    "REJECT": "not usable (FAIL)",
}
SUBJECT_LABELS = list(LABEL_RANK) + ["PARTIALLY_USABLE"]     # subject level: PARTIALLY_USABLE = some recordings usable, some rejected


def recording_label(overall: str, layers: dict[str, Any]) -> str:
    """Screening label of one recording. Only a FAIL is rejected; bad channels that can be interpolated are not."""
    if overall == "MISSING":
        return "MISSING"
    if overall == "FAIL":
        return "REJECT"
    if layers.get("integrity", {}).get("metrics", {}).get("interpolation_needed", False):
        return "USABLE_NEEDS_INTERPOLATION"
    return "USABLE_WITH_FLAGS" if overall == "FLAG" else "USABLE"


def recording_codes(info: dict[str, Any]) -> list[str]:
    """All reason codes of a recording (also the PASS notes such as USABLE_NEEDS_INTERPOLATION)."""
    if info["status"] == "ERROR":
        return ["FILE_NOT_READABLE"]
    if info["status"] == "MISSING":
        return ["RECORDING_MISSING"]
    return sorted({i["code"] for lay in info["layers"].values() for i in lay["issues"]})


def build_tables(st: QCState, sub: str, row: dict[str, str], sessions: list[str]):
    full_rows, short_rows = [], []
    for key in sorted(st.records, key=sort_key):
        info, L = st.records[key], st.records[key]["layers"]
        verdicts = {f"verdict_{l}": (L[l]["verdict"] if l in L else "N/A") for l in LAYER_ORDER}
        if info["status"] == "MISSING":
            overall = "MISSING"
        elif info["error"]:
            overall = "FAIL"
        else:
            overall = worst_verdict(verdicts.values())
        label = recording_label(overall, L)
        codes = "; ".join(recording_codes(info))
        ident = {"subject": key[0], "session": key[1], "task": key[2], "acq": key[3]}

        full = {**ident, "overall": overall, "label": label, "codes": codes, **verdicts, "load_error": info["error"] or ""}
        for l in LAYER_ORDER:
            if l in L:
                full.update(flatten_metrics(LAYER_PREFIX[l], L[l]["metrics"]))
        full.update({f"info_{k}": v for k, v in info["info"].items()})
        full_rows.append(full)

        r3 = lambda v: round(float(v), 2) if isinstance(v, (int, float, np.floating)) and not pd.isna(v) else v   # noqa: E731
        short_rows.append({
            **ident, "overall": overall, "label": label, "codes": codes, **verdicts,
            "duration_s": r3(metric(L, "header", "duration_s")),
            "n_bad_channels": metric(L, "integrity", "n_bad_channels"),
            "bad_channels": ", ".join(metric(L, "integrity", "bad_channels", [])),
            "bad_detail": metric(L, "integrity", "bad_detail", ""),
            "n_critical_bad": metric(L, "integrity", "n_critical_bad"),
            "critical_bad_channels": ", ".join(metric(L, "integrity", "critical_bad_channels", [])),
            "n_uncorrelated": metric(L, "integrity", "n_uncorrelated"),
            "uncorrelated_channels": ", ".join(metric(L, "integrity", "uncorrelated_channels", [])),
            "interpolation_needed": metric(L, "integrity", "interpolation_needed", ""),
            "median_spatial_corr": r3(metric(L, "integrity", "median_spatial_corr")),
            "frozen_s": metric(L, "integrity", "frozen_s"),
            "typical_amp_uv": r3(metric(L, "amplitude", "typical_amplitude_uv")),
            "median_kurtosis": r3(metric(L, "amplitude", "median_kurtosis")),
            "alpha_peak_hz": r3(metric(L, "spectral", "alpha_peak_hz")),
            "alpha_prom_db": r3(metric(L, "spectral", "alpha_prominence_db")),
            "alpha_ratio_EC_EO": r3(metric(L, "spectral", "alpha_ratio_EC_EO")),
            "alpha_non_responder_possible": metric(L, "spectral", "alpha_non_responder_possible", ""),
            "line50_db": r3(metric(L, "spectral", "line50_db")),
            "line100_db": r3(metric(L, "spectral", "line100_db")),
            "line150_db": r3(metric(L, "spectral", "line150_db")),
            "notch_suggested": metric(L, "spectral", "notch_suggested", ""),
            "trigger_events": info["info"].get("stim_n_events", np.nan),
            "load_error": info["error"] or "",
        })

    df_full, df_rec = pd.DataFrame(full_rows), pd.DataFrame(short_rows)

    all_bad = [c for s in df_rec["bad_channels"] for c in str(s).split(", ") if c]
    labels = [lab for lab in df_rec["label"] if lab in LABEL_RANK]
    if not labels:
        subject_label = "NO DATA"
    elif all(lab == "REJECT" for lab in labels):
        subject_label = "REJECT"                                   # nothing usable
    elif any(lab == "REJECT" for lab in labels):
        subject_label = "PARTIALLY_USABLE"                         # some recordings are usable, others are rejected
    else:
        subject_label = max(labels, key=LABEL_RANK.__getitem__)    # the most demanding label of the usable recordings
    df_sub = pd.DataFrame([{
        "subject": sub, "sex": row["sex"], "age": row["age"], "sessions_analysed": ", ".join(sessions),
        "n_recordings": len(df_rec),
        "n_PASS": int((df_rec["overall"] == "PASS").sum()), "n_FLAG": int((df_rec["overall"] == "FLAG").sum()),
        "n_FAIL": int((df_rec["overall"] == "FAIL").sum()), "n_MISSING": int((df_rec["overall"] == "MISSING").sum()),
        "n_USABLE": int((df_rec["label"] == "USABLE").sum()),
        "n_USABLE_WITH_FLAGS": int((df_rec["label"] == "USABLE_WITH_FLAGS").sum()),
        "n_USABLE_NEEDS_INTERPOLATION": int((df_rec["label"] == "USABLE_NEEDS_INTERPOLATION").sum()),
        "n_REJECT": int((df_rec["label"] == "REJECT").sum()),
        "overall": worst_verdict(df_rec["overall"], empty="NO DATA"),
        "label": subject_label,
        "channels_flagged_bad (count of recordings)": ", ".join(f"{k} x{v}" for k, v in Counter(all_bad).most_common(6)),
    }])

    parts = []
    for key, t in st.ch_tables.items():
        part = t.copy()
        part.insert(0, "acq", key[3]); part.insert(0, "task", key[2])
        part.insert(0, "session", key[1]); part.insert(0, "subject", key[0])
        parts.append(part)
    df_ch = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df_full, df_rec, df_sub, df_ch


def build_issues(st: QCState) -> pd.DataFrame:
    """Every finding that is not a plain PASS (FAIL, FLAG, MISSING) plus the PASS notes with a code
    (for example USABLE_NEEDS_INTERPOLATION), worst first."""
    rows = []
    for key in sorted(st.records, key=sort_key):
        sub, ses, task, acq = key
        info = st.records[key]
        base = {"subject": sub, "session": ses, "task": task, "acq": acq}
        if info["status"] == "MISSING":
            rows.append({**base, "layer": "load", "code": "RECORDING_MISSING", "finding": "recording_missing",
                         "value": info["error"], "limit": "recording expected for a completed session", "verdict": "MISSING"})
            continue
        if info["error"]:
            rows.append({**base, "layer": "load", "code": "FILE_NOT_READABLE", "finding": "file_not_readable",
                         "value": info["error"], "limit": "file must be readable", "verdict": "FAIL"})
            continue
        for layer in LAYER_ORDER:
            for i in info["layers"].get(layer, {}).get("issues", []):
                rows.append({**base, "layer": layer, "code": i["code"], "finding": i["metric"], "value": i["value"],
                             "limit": i["threshold"], "verdict": i["verdict"]})
    df = pd.DataFrame(rows, columns=["subject", "session", "task", "acq", "layer", "code", "finding", "value", "limit", "verdict"])
    order = {"FAIL": 0, "MISSING": 1, "FLAG": 2, "PASS": 3}
    return df.sort_values("verdict", key=lambda s: s.map(lambda v: order.get(v, 9)), kind="stable").reset_index(drop=True)


# ======================================================================
# FIGURES (pilot notebook, cells 5, 9 and 10)
# ======================================================================
def representative_key(st: QCState, sub: str) -> tuple | None:
    """The recording shown in the trace and PSD-image figures: ses-1, Eyes Closed, pre (or the first one available)."""
    keys = sorted((k for k in st.ch_tables if k[0] == sub), key=sort_key)
    wanted = (sub, "ses-1", "EyesClosed", "pre")
    return wanted if wanted in st.ch_tables else (keys[0] if keys else None)


def plot_psd_subject(st: QCState, sub: str, fig_dir: Path) -> None:
    """Figure A: PSD of O1/O2/Oz, Eyes Closed vs Eyes Open. Rows = (session, acq), columns = channels."""
    combos = sorted({(k[1], k[3]) for k in st.psd if k[0] == sub}, key=lambda x: (x[0], ACQS.index(x[1])))
    chans = CONFIG["psd_channels"]
    if not combos:
        return
    fig, axes = plt.subplots(len(combos), len(chans), figsize=(5 * len(chans), 3.0 * len(combos)), squeeze=False, sharex=True)
    for r, (ses, acq) in enumerate(combos):
        for c_i, ch in enumerate(chans):
            ax = axes[r][c_i]
            for task, color in (("EyesClosed", "tab:blue"), ("EyesOpen", "tab:orange")):
                psd = st.psd.get((sub, ses, task, acq))
                i = find_channel(psd["channels"], ch) if psd else None
                if i is None:
                    continue
                f = psd["freqs"]; m = (f >= 1) & (f <= 120)
                ax.semilogy(f[m], psd["psd"][i][m], color=color, lw=1, label=task.replace("Eyes", "Eyes "))
            ax.axvspan(*CONFIG["alpha_band_hz"], color="green", alpha=0.15, label="alpha 8-12 Hz")
            ax.axvline(CONFIG["line_hz"], color="k", ls="--", lw=0.8, label="50 Hz")
            for h in CONFIG["harmonics_hz"]:
                ax.axvline(h, color="gray", ls=":", lw=0.8)
            ax.set_title(f"{ses} {acq} - {ch}", fontsize=10)
            if c_i == 0:
                ax.set_ylabel("PSD (uV$^2$/Hz)")
            if r == len(combos) - 1:
                ax.set_xlabel("Frequency (Hz)")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{sub}: alpha peak and alpha blocking (Eyes Closed vs Eyes Open), Welch PSD", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / f"A_psd_occipital_{sub}.png", dpi=130, bbox_inches="tight")


def plot_amplitude_heatmap(df_channels: pd.DataFrame, sub: str, fig_dir: Path) -> None:
    """Figure B: typical amplitude of every channel in every recording (log colours). 'x' = channel flagged bad."""
    d = df_channels[df_channels["subject"] == sub]
    if d.empty:
        return
    order = list(dict.fromkeys(d["channel"]))
    combos = sorted(set(zip(d["session"], d["task"], d["acq"])), key=lambda c: (c[0], TASKS.index(c[1]), ACQS.index(c[2])))
    labels = [f"{s.replace('ses-', 's')}-{'EC' if t == 'EyesClosed' else 'EO'}-{a}" for s, t, a in combos]
    cols = ["session", "task", "acq"]
    amp = d.pivot_table(index="channel", columns=cols, values="window_std_uv", aggfunc="first").reindex(order)[combos]
    bad = d.pivot_table(index="channel", columns=cols, values="bad", aggfunc="first").reindex(order)[combos]
    vals = np.ma.masked_invalid(np.clip(amp.to_numpy(dtype=float), 0.1, None))
    bad_arr = bad.apply(lambda col: col.map(lambda x: bool(x) if pd.notna(x) else False)).to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(1.0 * len(combos) + 3, 0.19 * len(order) + 2))
    im = ax.imshow(vals, aspect="auto", cmap="viridis", norm=LogNorm(vmin=max(vals.min(), 0.5), vmax=max(vals.max(), 10)))
    yy, xx = np.where(bad_arr)
    ax.scatter(xx, yy, marker="x", color="red", s=40, label="flagged bad (flat, clipped or noisy)")
    ax.set_xticks(range(len(combos))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
    fig.colorbar(im, ax=ax, label="typical 1-s STD (uV, log scale)")
    if len(yy):
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"{sub}: channel amplitude per recording (EC = Eyes Closed, EO = Eyes Open)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"B_amplitude_heatmap_{sub}.png", dpi=130, bbox_inches="tight")


def plot_traces(st: QCState, sub: str, fig_dir: Path) -> None:
    """Figure C: visual trace inspection. Left: all channels for 6 s. Right: typical amplitude per channel."""
    key = representative_key(st, sub)
    if key is None or key not in st.excerpts:
        return
    seg, t = st.excerpts[key], st.ch_tables[key]
    sfreq = float(st.summaries[key]["sfreq"])
    names = t["channel"].tolist()
    n_ch = len(names)
    typical = float(np.median(t["window_std_uv"]))
    spacing = max(4.0 * typical, 20.0)
    time_ax = np.arange(seg.shape[1]) / sfreq
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 0.19 * n_ch + 2.5), gridspec_kw={"width_ratios": [3, 1]})
    for i in range(n_ch):
        ax1.plot(time_ax, seg[i] + (n_ch - 1 - i) * spacing, lw=0.5, color="tab:red" if t["bad"].iloc[i] else "k")
    ax1.set_yticks([(n_ch - 1 - i) * spacing for i in range(n_ch)]); ax1.set_yticklabels(names, fontsize=6)
    ax1.set_ylim(-3.2 * spacing, n_ch * spacing); ax1.set_xlim(0, time_ax[-1])
    y0 = -2.8 * spacing                                          # scale bar: 100 uV
    ax1.plot([time_ax[-1] * 0.985] * 2, [y0, y0 + 100], color="b", lw=3)
    ax1.text(time_ax[-1] * 0.97, y0 + 50, "100 uV", ha="right", va="center", fontsize=9, color="b",
             bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"{key[0]} {key[1]} {key[2]} {key[3]}: 6 s of EEG, all channels "
                  "(baseline-corrected for display; red = flagged bad)", fontsize=10)
    ypos = np.arange(n_ch)[::-1]
    ax2.barh(ypos, t["window_std_uv"].clip(lower=0.01), color=["tab:red" if b_ else "tab:blue" for b_ in t["bad"]])
    ax2.axvline(CONFIG["typical_amp_max_uv"], color="k", ls="--", lw=0.8, label=f"limit {CONFIG['typical_amp_max_uv']:.0f} uV")
    ax2.set_xscale("log"); ax2.set_xlim(0.01, max(300.0, float(t["window_std_uv"].max()) * 1.5))
    ax2.set_ylim(-0.5, n_ch - 0.5)
    ax2.set_yticks(ypos); ax2.set_yticklabels(names, fontsize=6)
    ax2.set_xlabel("typical 1-s STD (uV, log)"); ax2.set_title("Typical amplitude per channel", fontsize=10)
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"C_traces_{sub}.png", dpi=130, bbox_inches="tight")


def plot_psd_image(st: QCState, sub: str, fig_dir: Path) -> None:
    """Figure D: PSD of ALL channels (dB). A bad channel is a bright or dark horizontal band; line noise is a vertical stripe."""
    key = representative_key(st, sub)
    if key is None or key not in st.psd:
        return
    psd = st.psd[key]
    f = psd["freqs"]; m = (f >= 1) & (f <= 120)
    db = 10 * np.log10(np.clip(psd["psd"][:, m].astype(float), 1e-12, None))
    fig = plt.figure(figsize=(12, 0.17 * db.shape[0] + 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[1, 4], hspace=0.08, wspace=0.03)
    ax0, ax1, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0], sharex=None), fig.add_subplot(gs[1, 1])
    ax0.plot(f[m], np.median(db, axis=0), "k", lw=1, label="median over channels")
    ax0.fill_between(f[m], np.percentile(db, 5, axis=0), np.percentile(db, 95, axis=0), color="gray", alpha=0.3,
                     label="5-95% of channels")
    ax0.axvspan(*CONFIG["alpha_band_hz"], color="green", alpha=0.15); ax0.axvline(CONFIG["line_hz"], color="r", ls="--", lw=0.8)
    ax0.set_xlim(f[m][0], f[m][-1]); ax0.set_ylabel("dB"); ax0.legend(fontsize=7, loc="upper right")
    ax0.set_title(f"{key[0]} {key[1]} {key[2]} {key[3]}: power spectrum of all channels (green = alpha, red = 50 Hz)")
    ax0.tick_params(labelbottom=False)
    im = ax1.imshow(db, aspect="auto", cmap="magma", extent=[f[m][0], f[m][-1], db.shape[0], 0])
    ax1.set_yticks(np.arange(db.shape[0]) + 0.5); ax1.set_yticklabels(psd["channels"], fontsize=6)
    ax1.set_xlabel("Frequency (Hz)")
    fig.colorbar(im, cax=cax, label="10*log10(uV$^2$/Hz)")
    fig.savefig(fig_dir / f"D_psd_all_channels_{sub}.png", dpi=130, bbox_inches="tight")


def plot_overview(key: tuple, ov: dict, psd: dict, fig_dir: Path) -> None:
    """Figure F (one per recording): whole recording on O1/O2/Oz (frozen segments shaded red) and the PSD."""
    sub, ses, task, acq = key
    if not ov["traces"]:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5))
    vals = []
    for target, sig in ov["traces"].items():
        t_plot = np.arange(len(sig)) * ov["step"] / ov["sfreq"]
        axes[0].plot(t_plot, sig, lw=0.35, alpha=0.85, label=target)
        vals.append(sig)
    for s0, length in ov["frozen_segs"]:
        axes[0].axvspan(s0, s0 + length, color="red", alpha=0.25)
    y_min, y_max = np.percentile(np.concatenate(vals), [0.5, 99.5])
    span = max(y_max - y_min, 50.0)
    center = 0.5 * (y_max + y_min)
    axes[0].set_ylim(center - 0.85 * span, center + 0.85 * span)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("uV (median removed)")
    axes[0].set_title(f"{sub} {ses} {task} {acq}: whole recording on the occipital channels (red = all channels frozen)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(True, alpha=0.2, ls="--")

    f = psd["freqs"]
    m = (f >= 1) & (f <= 120)
    for target in CONFIG["psd_channels"]:
        i = find_channel(psd["channels"], target)
        if i is not None:
            axes[1].semilogy(f[m], psd["psd"][i][m], lw=0.8, label=target)
    axes[1].axvspan(*CONFIG["alpha_band_hz"], color="green", alpha=0.15, label="alpha 8-12 Hz")
    axes[1].axvline(CONFIG["line_hz"], color="k", ls="--", lw=0.8, label="50 Hz")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (uV$^2$/Hz)")
    axes[1].set_title(f"Welch PSD, {task} ({sub} {ses} {acq})")
    axes[1].legend(loc="upper right", ncol=5, fontsize=8)
    axes[1].grid(True, which="both", alpha=0.15, ls="--")
    fig.tight_layout()
    fig.savefig(fig_dir / f"F_overview_{sub}_{ses}_{task}_{acq}.png", dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------- 3D scalp maps (Figure E)
def scalp_positions(ch_names: list[str]):
    """3D electrode positions from MNE's standard templates (display only).

    The standard 10-20 template is used where it knows the channel; the 10-05 template (a superset: the 64-channel
    layout contains many 10-10 names such as AF7 or PO3) fills in the rest. Both live in the same coordinate space.
    Normalizes MNI scalp coordinates onto a centered unit-radius head sphere (r = 0.095 m)
    so that electrodes align accurately with the schematic head in both top and side views.
    Returns (found_names, xyz array in metres, missing_names). Raises if the templates cannot be loaded.
    """
    pos_1005 = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    try:
        pos_1020 = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    except Exception:
        pos_1020 = {}

    # Fit an axis-aligned ellipsoid center & radii to the full 10-05 template
    # because MNE's standard_1005 is in MNI space (not centered at 0,0,0 and elongated in Y)
    all_pts = np.array([v for v in pos_1005.values() if np.isfinite(v).all()], dtype=float)
    center = np.array([
        0.0,                                                     # X is symmetric
        0.5 * (np.percentile(all_pts[:, 1], 99) + np.percentile(all_pts[:, 1], 1)),  # Y center shift (~ -16 mm)
        np.percentile(all_pts[:, 2], 12),                        # Z base (inion/nasion equator plane)
    ])
    shifted_all = all_pts - center
    radii = np.array([
        np.percentile(np.abs(shifted_all[:, 0]), 98),
        np.percentile(np.abs(shifted_all[:, 1]), 98),
        np.percentile(shifted_all[:, 2], 99),
    ])
    head_r = 0.095  # standard schematic head radius in metres (9.5 cm)

    lut = {}
    for source in (pos_1005, pos_1020):                          # the 10-20 template is read last, so it wins
        for k, v in source.items():
            v = np.asarray(v, dtype=float)
            if np.isfinite(v).all():
                u = (v - center) / radii                         # map from the MNI ellipsoid to a centered spherical shell
                norm = np.linalg.norm(u)
                if norm > 0:
                    u = u / norm * head_r
                lut[k.lower()] = u

    found, xyz, missing = [], [], []
    for n in ch_names:
        v = lut.get(n.lower())
        if v is None:
            missing.append(n)
        else:
            found.append(n)
            xyz.append(v)
    return found, np.array(xyz), missing


def draw_head(ax, r: float, view: str) -> None:
    """A simple head: outline, wire sphere (upper part), nose (front = +y) and ears."""
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi * 0.58, 16)
    ax.plot_wireframe(
        r * np.outer(np.cos(u), np.sin(v)),
        r * np.outer(np.sin(u), np.sin(v)),
        r * np.outer(np.ones_like(u), np.cos(v)),
        color="0.82", lw=0.35, alpha=0.55, zorder=1
    )
    t = np.linspace(0, 2 * np.pi, 80)
    if view == "top":
        # Head equatorial outline
        ax.plot(r * np.cos(t), r * np.sin(t), np.zeros_like(t), color="k", lw=1.0, zorder=2)
        # Nose in top view (pointing +Y)
        ax.plot([-0.11 * r, 0, 0.11 * r], [0.99 * r, 1.15 * r, 0.99 * r], [0, 0, 0], color="k", lw=1.4, zorder=2)
        # Ears on both sides
        ear_t = np.linspace(-np.pi / 2, np.pi / 2, 30)
        for sgn in (-1, 1):
            ax.plot(sgn * (1.0 * r + 0.04 * r * np.cos(ear_t)), 0.18 * r * np.sin(ear_t), np.zeros_like(ear_t), color="k", lw=1.1, zorder=2)
    else:
        # Nose in side view (pointing +Y / left of image, with bridge of nose)
        ax.plot([0, 0, 0], [0.98 * r, 1.14 * r, 0.99 * r], [0.12 * r, -0.02 * r, -0.08 * r], color="k", lw=1.4, zorder=2)
        # Left ear outline
        ear_t = np.linspace(0, 2 * np.pi, 40)
        ax.plot(np.full_like(ear_t, -1.0 * r), -0.03 * r + 0.11 * r * np.cos(ear_t), -0.06 * r + 0.20 * r * np.sin(ear_t), color="k", lw=1.1, zorder=2)


def scalp_panel(ax, xyz, values, cmap, norm, view, labels=None, bad_mask=None):
    """One 3D panel: electrodes coloured by `values`. view = 'top' (nose up) or 'side' (left side)."""
    # Use orthographic projection to eliminate 3D perspective distortion
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    r = float(np.median(np.linalg.norm(xyz, axis=1)))

    if view == "top":
        # Azimuthal equidistant-style flattening on XY plane for crisp top view, while keeping Z for layering
        xy_norm = np.linalg.norm(xyz[:, :2], axis=1, keepdims=True)
        xy_dir = np.divide(xyz[:, :2], xy_norm, out=np.zeros_like(xyz[:, :2]), where=xy_norm > 1e-9)
        # Theta from zenith (0 at Cz, pi/2 at equator)
        cos_theta = np.clip(xyz[:, 2] / np.linalg.norm(xyz, axis=1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        # Map 0..pi*0.55 onto 0..r so peripheral electrodes sit neatly inside/on the head circle
        rho = (theta / (0.55 * np.pi)) * r
        plot_xyz = np.column_stack([xy_dir[:, 0] * rho, xy_dir[:, 1] * rho, xyz[:, 2]])
    else:
        plot_xyz = xyz.copy()

    draw_head(ax, r, view)

    sc = ax.scatter(
        plot_xyz[:, 0], plot_xyz[:, 1], plot_xyz[:, 2],
        c=values, cmap=cmap, norm=norm, s=75,
        edgecolors="k", linewidths=0.45, depthshade=False, zorder=4
    )
    if bad_mask is not None and bad_mask.any():
        ax.scatter(
            *plot_xyz[bad_mask].T, s=190, facecolors="none",
            edgecolors="red", linewidths=1.6, depthshade=False, zorder=5
        )
    if labels is not None:
        for (x_, y_, z_), name in zip(plot_xyz, labels):
            # Offset slightly toward +Y in top view so text appears above each marker
            ax.text(
                x_, y_ + (0.065 * r if view == "top" else 0.0), z_ + 0.05 * r,
                name, fontsize=5.5, ha="center", va="center", zorder=6
            )

    lim = 1.25 * r
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.30 * r, 1.15 * r)
    try:
        ax.set_box_aspect((2 * lim, 2 * lim, 1.45 * r), zoom=1.12)
    except TypeError:  # older matplotlib without the zoom option
        ax.set_box_aspect((2 * lim, 2 * lim, 1.45 * r))

    # top view: nose up (+Y up, +X right). side view: left side of the head (-X towards viewer, +Y nose to the left)
    if view == "top":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=8, azim=180)
    ax.set_axis_off()
    return sc


def per_channel_maps(df_channels: pd.DataFrame, sub: str) -> dict[str, pd.Series]:
    """Median over the recordings of the subject, per channel."""
    d = df_channels[df_channels["subject"] == sub]
    order = list(dict.fromkeys(d["channel"]))
    amp = d.groupby("channel")["window_std_uv"].median().reindex(order)
    line = d.groupby("channel")["line50_db"].median().reindex(order)
    badfrac = d.groupby("channel")["bad"].mean().reindex(order)
    # alpha blocking per channel: alpha power Eyes Closed / Eyes Open, same session + acq
    pw = d.pivot_table(index=["subject", "session", "acq", "channel"], columns="task", values="alpha_power_uv2", aggfunc="first")
    if {"EyesClosed", "EyesOpen"} <= set(pw.columns):
        ratio = (pw["EyesClosed"] / pw["EyesOpen"]).groupby("channel").median().reindex(order)
    else:
        ratio = pd.Series(np.nan, index=order)
    return {"amp": amp, "ratio": ratio, "line": line, "bad": badfrac}


def plot_scalp_3d(df_channels: pd.DataFrame, sub: str, pos, fig_dir: Path) -> None:
    """Figure E: four scalp maps (top view above, side view below)."""
    found, xyz, missing = pos
    maps = per_channel_maps(df_channels, sub)
    idx = [c for c in found if c in maps["amp"].index]
    xyz_s = np.array([xyz[found.index(c)] for c in idx])
    a, rt, ln, bd = (maps[k].loc[idx].to_numpy(dtype=float) for k in ("amp", "ratio", "line", "bad"))
    amp_hi = max(30.0, float(np.nanpercentile(a, 98)) * 1.2)
    panels = [
        ("Typical amplitude (uV)\n1-s STD", a, "viridis", Normalize(0, amp_hi)),
        ("Alpha blocking\npower Eyes Closed / Eyes Open", rt, "plasma", LogNorm(1, 8)),
        ("50 Hz line noise (dB)\nabove neighbouring frequencies", ln, "inferno", Normalize(0, 30)),
        ("Share of recordings in which\nthe channel was flagged bad", bd, "Reds", Normalize(0, 1)),
    ]
    bad_mask = np.nan_to_num(bd) >= 0.5
    fig = plt.figure(figsize=(21, 10))
    tops = []
    for j, (title, vals, cmap, norm) in enumerate(panels):
        for row, view in enumerate(("top", "side")):
            ax = fig.add_subplot(2, 4, row * 4 + j + 1, projection="3d")
            sc = scalp_panel(
                ax, xyz_s, vals, cmap, norm, view,
                labels=idx if (view == "top" and j == 0) else None,
                bad_mask=bad_mask if j == 3 else None
            )
            if row == 0:
                ax.set_title(title, fontsize=11)
                tops.append((ax, sc, norm))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=-0.12, hspace=-0.22)
    for ax, sc, norm in tops:  # one colour bar under each column
        pos_ax = ax.get_position()
        cax = fig.add_axes([pos_ax.x0 + 0.035, 0.06, pos_ax.width - 0.07, 0.015])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        if isinstance(norm, LogNorm):
            cb.set_ticks([1, 2, 4, 8])
            cb.set_ticklabels(["1", "2", "4", "8"])
            cb.ax.minorticks_off()
    fig.suptitle(
        f"{sub}: scalp maps on the standard 10-05 electrode template "
        "(top row: view from above, nose up; bottom row: left side)",
        y=0.97, fontsize=13
    )
    fig.savefig(fig_dir / f"E_scalp_3d_{sub}.png", dpi=130, bbox_inches="tight")
    return f"not in the electrode template, not drawn: {', '.join(missing)}" if missing else None


CLASS_STYLE = {                       # channel quality classes of the topomap: (colour, legend text)
    "GOOD": ("#4caf50", "Green: good / normal"),
    "NOISY": ("#ffd92f", "Yellow: noisy or poor spatial correlation"),
    "LINE_DRIFT": ("#ff8c00", "Orange: high 50 Hz line noise or voltage drift"),
    "BAD": ("#d62728", "Red: bad / dead / flat (or clipped) channel"),
}
CRIT_RED = "#b00020"
VERDICT_TEXT_COLOR = {"PASS": "#2e7d32", "FLAG": "#b26a00", "FAIL": "#c62828", "MISSING": "#6a1b9a"}


def channel_class(r) -> str:
    """Colour class of ONE channel in ONE recording (priority: red > yellow > orange > green)."""
    if r["flat"] or r["clipped"]:
        return "BAD"
    if r["noisy"] or r["uncorrelated"]:
        return "NOISY"
    if (r["line50_db"] > CONFIG["line_flag_db"]) or (abs(r["drift_uv"]) > CONFIG["drift_info_uv"]):
        return "LINE_DRIFT"
    return "GOOD"


def project_top(xyz: np.ndarray) -> np.ndarray:
    """3D electrode positions -> 2D top view (nose up). The head circle (equator) has radius 1."""
    xy_norm = np.linalg.norm(xyz[:, :2], axis=1, keepdims=True)
    xy_dir = np.divide(xyz[:, :2], xy_norm, out=np.zeros_like(xyz[:, :2]), where=xy_norm > 1e-9)
    theta = np.arccos(np.clip(xyz[:, 2] / np.linalg.norm(xyz, axis=1), -1.0, 1.0))     # angle from the vertex (Cz)
    return xy_dir * (theta / (0.55 * np.pi))[:, None]


def _draw_head_2d(ax) -> None:
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.6, color="k", zorder=1))
    ax.plot([-0.13, 0, 0.13], [0.985, 1.13, 0.985], color="k", lw=1.6, zorder=1)          # nose (front = up)
    t = np.linspace(-np.pi / 2, np.pi / 2, 30)
    for s in (-1, 1):                                                                       # ears
        ax.plot(s * (1.0 + 0.07 * np.cos(t)), 0.16 * np.sin(t), color="k", lw=1.4, zorder=1)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.2, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_channel_quality_topomap(st: QCState, df_rec: pd.DataFrame, df_channels: pd.DataFrame, pos, sub: str,
                                 fig_dir: Path) -> str | None:
    """Channel Quality Topomap: one head per recording, every channel coloured by its quality.

    Green = good | yellow = noisy / poor spatial correlation | orange = high 50 Hz line noise or voltage drift |
    red = bad / dead / flat (or clipped). A CRITICAL channel (Cz, Fz, Pz, O1, O2) that is bad gets a thick red
    square and a red star. Saved as <sub>_task-rest_channel_quality_topomap.png
    """
    from matplotlib.lines import Line2D
    found, xyz, missing = pos
    xy = project_top(xyz)
    idx = {n: i for i, n in enumerate(found)}
    d = df_channels[df_channels["subject"] == sub]
    sessions = sorted({k[1] for k in st.records})
    cols = [(t, a) for t in TASKS for a in ACQS]
    fig, axes = plt.subplots(len(sessions), len(cols), figsize=(15.6, 4.0 * len(sessions) + 2.0), squeeze=False)
    for i, ses in enumerate(sessions):
        for j, (task, acq) in enumerate(cols):
            ax = axes[i][j]
            _draw_head_2d(ax)
            t = d[(d["session"] == ses) & (d["task"] == task) & (d["acq"] == acq)]
            rr = df_rec[(df_rec["session"] == ses) & (df_rec["task"] == task) & (df_rec["acq"] == acq)]
            short = f"{ses}  {'Eyes Closed' if task == 'EyesClosed' else 'Eyes Open'}  {acq}"
            overall = str(rr.iloc[0]["overall"]) if len(rr) else "MISSING"
            if t.empty:
                ax.set_title(f"{short}\n{overall}", fontsize=9.5, color=VERDICT_TEXT_COLOR.get(overall, "k"))
                ax.text(0, 0, "no data", ha="center", va="center", fontsize=11, color="0.45")
                continue
            for _, r in t.iterrows():
                if r["channel"] not in idx:
                    continue
                x, y = xy[idx[r["channel"]]]
                ax.scatter([x], [y], s=185, c=CLASS_STYLE[channel_class(r)][0], edgecolors="k", linewidths=0.6, zorder=3)
                ax.text(x, y, r["channel"], fontsize=4.6, ha="center", va="center", zorder=5,
                        fontweight="bold" if r["is_critical"] else "normal")
                if r["is_critical"] and r["bad"]:                     # warning marker for a bad CRITICAL channel
                    ax.scatter([x], [y], s=520, marker="s", facecolors="none", edgecolors=CRIT_RED, linewidths=2.8, zorder=4)
                    ax.scatter([x + 0.09], [y + 0.09], s=140, marker="*", c=CRIT_RED, edgecolors="white", linewidths=0.6, zorder=6)
            label = str(rr.iloc[0]["label"]) if len(rr) else ""
            n_bad = int(t["bad"].sum())
            ax.set_title(f"{short}\n{overall}  |  {label}  |  {n_bad} bad", fontsize=8.6,
                         color=VERDICT_TEXT_COLOR.get(overall, "k"))

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k", markersize=11, label=txt)
               for c, txt in CLASS_STYLE.values()]
    handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor=CRIT_RED,
                          markeredgewidth=2.6, markersize=13,
                          label=f"Critical channel bad ({', '.join(CONFIG['critical_channels'])}): square + star"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    c = CONFIG
    fig.suptitle(f"{sub}: channel quality topomap (top view, nose up; standard 10-20 template, 10-05 for the other names)",
                 fontsize=12.5, y=0.995)
    fig.text(0.5, 0.075,
             f"Yellow: typical 1-s STD > {c['noisy_ratio']:.0f}x the median and > {c['noisy_min_uv']:.0f} uV, or |r| with the median of all "
             f"channels < {c['corr_threshold']}.   Orange: 50 Hz > {c['line_flag_db']:.0f} dB above its flanks, or |drift| > "
             f"{c['drift_info_uv']:.0f} uV.   Red: flat (STD < {c['flat_std_uv']} uV) or clipped.",
             ha="center", fontsize=8, color="0.3")
    if missing:
        fig.text(0.5, 0.055, f"Not in the electrode template (not drawn): {', '.join(missing)}", ha="center", fontsize=8, color="0.3")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.14, wspace=0.02, hspace=0.18)
    fig.savefig(fig_dir / f"{sub}_task-rest_channel_quality_topomap.png", dpi=140, bbox_inches="tight")
    return None


def plot_spatial_correlation(st: QCState, df_channels: pd.DataFrame, sub: str, fig_dir: Path) -> None:
    """Figure G: how strongly every channel is correlated with the median of all channels.

    Left: median |r| per channel and recording (red x = below the threshold). Middle: the same for the
    representative recording, sorted (a channel far to the left is an uncorrelated / noisy channel).
    Right: |r| in every 5-s window of the representative recording (a channel that is bad only part of the time
    shows up as a dark stripe).
    """
    d = df_channels[df_channels["subject"] == sub]
    key = representative_key(st, sub)
    if d.empty or key is None or key not in st.corr_windows:
        return
    thr = CONFIG["corr_threshold"]
    order = list(dict.fromkeys(d["channel"]))
    combos = sorted(set(zip(d["session"], d["task"], d["acq"])), key=lambda c: (c[0], TASKS.index(c[1]), ACQS.index(c[2])))
    labels = [f"{s.replace('ses-', 's')}-{'EC' if t == 'EyesClosed' else 'EO'}-{a}" for s, t, a in combos]
    r_tab = d.pivot_table(index="channel", columns=["session", "task", "acq"], values="corr_median",
                          aggfunc="first").reindex(order)[combos]
    rv = r_tab.to_numpy(dtype=float)
    t = st.ch_tables[key]
    names = t["channel"].tolist()
    win, wsec = st.corr_windows[key], st.corr_window_s[key]
    n_ch = len(names)

    fig = plt.figure(figsize=(18, 0.21 * n_ch + 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.55 * len(combos) + 1.0, 1.8, 3.0, 0.07], wspace=0.3, top=0.93, bottom=0.07)
    ax1, ax2, ax3, cax = (fig.add_subplot(gs[0, i]) for i in range(4))

    im = ax1.imshow(rv, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    yy, xx = np.where(rv < thr)
    if len(yy):
        ax1.scatter(xx, yy, marker="x", color="red", s=34, label=f"|r| < {thr}")
        ax1.legend(loc="upper right", fontsize=7)
    ax1.set_xticks(range(len(combos)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order, fontsize=6)
    ax1.set_title("Median |r| per recording", fontsize=10)

    vals = t["corr_median"].to_numpy(dtype=float)
    srt = np.argsort(vals)
    colors = ["0.6" if (t["flat"].iloc[i] or t["clipped"].iloc[i]) else ("#d62728" if t["uncorrelated"].iloc[i] else "#4c78a8")
              for i in srt]
    ax2.barh(np.arange(n_ch), vals[srt], color=colors)
    ax2.axvline(thr, color="k", ls="--", lw=1.0, label=f"threshold {thr}")
    ax2.set_yticks(np.arange(n_ch))
    ax2.set_yticklabels([names[i] for i in srt], fontsize=6)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.6, n_ch - 0.4)
    ax2.set_xlabel("median |r| over windows")
    ax2.set_title(f"{key[1]} {key[2]} {key[3]}, sorted\n(red = uncorrelated, grey = flat / clipped)", fontsize=10)
    ax2.legend(loc="lower right", fontsize=7)

    ax3.imshow(win, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=[0, win.shape[1] * wsec, n_ch, 0])
    ax3.set_yticks(np.arange(n_ch) + 0.5)
    ax3.set_yticklabels(names, fontsize=6)
    ax3.set_xlabel("time (s)")
    ax3.set_title(f"|r| in every {wsec:.0f}-s window", fontsize=10)
    fig.colorbar(im, cax=cax, label="|Pearson r| with the median of all channels")
    fig.suptitle(f"{sub}: spatial correlation of every channel with the median of all channels "
                 f"(baseline-corrected, {CONFIG['corr_band_hz'][0]:.0f}-{CONFIG['corr_band_hz'][1]:.0f} Hz). "
                 f"Channels below |r| = {thr} are counted as bad (uncorrelated / noisy).", fontsize=11, y=0.985)
    fig.savefig(fig_dir / f"G_spatial_correlation_{sub}.png", dpi=130, bbox_inches="tight")


def make_figures(st: QCState, sub: str, df_rec: pd.DataFrame, df_channels: pd.DataFrame, fig_dir: Path) -> dict[str, list[str]]:
    """Draw and save all figures. One failing figure never stops the others.

    Nothing is printed here (a spinner is running). Returns {"saved": [...], "errors": [...], "notes": [...]}.
    """
    errors: list[str] = []
    notes: list[str] = []

    def safe(name: str, fn, *a) -> None:
        try:
            res = fn(*a)
            if isinstance(res, str):
                notes.append(f"{name}: {res}")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
        finally:
            plt.close("all")

    safe("A_psd_occipital", plot_psd_subject, st, sub, fig_dir)
    safe("B_amplitude_heatmap", plot_amplitude_heatmap, df_channels, sub, fig_dir)
    safe("C_traces", plot_traces, st, sub, fig_dir)
    safe("D_psd_all_channels", plot_psd_image, st, sub, fig_dir)
    safe("G_spatial_correlation", plot_spatial_correlation, st, df_channels, sub, fig_dir)
    try:
        pos = scalp_positions(df_channels["channel"].drop_duplicates().tolist())
    except Exception as e:
        pos = None
        errors.append(f"E_scalp_3d and channel_quality_topomap: electrode templates not available "
                      f"({type(e).__name__}: {e}); figures skipped")
    if pos is not None:
        safe("E_scalp_3d", plot_scalp_3d, df_channels, sub, pos, fig_dir)
        safe("channel_quality_topomap", plot_channel_quality_topomap, st, df_rec, df_channels, pos, sub, fig_dir)
    for key in sorted(st.overviews, key=sort_key):
        safe(f"F_overview_{key[1]}_{key[2]}_{key[3]}", plot_overview, key, st.overviews[key], st.psd[key], fig_dir)
    return {"saved": sorted(p.name for p in fig_dir.glob("*.png")), "errors": errors, "notes": notes}


# ======================================================================
# REPORT
# ======================================================================
class Report:
    """Collects the lines of qc_report.txt (the console only shows a short summary)."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna()


def key_finding_lines(df_rec: pd.DataFrame, df_ch: pd.DataFrame) -> list[str]:
    """The important numbers of a subject as plain sentences (used in the report and in status.txt)."""
    out: list[str] = []
    counts = df_rec["overall"].value_counts().to_dict()
    out.append(f"Recordings: {len(df_rec)} expected  |  PASS {counts.get('PASS', 0)}  |  FLAG {counts.get('FLAG', 0)}  "
               f"|  FAIL {counts.get('FAIL', 0)}  |  MISSING {counts.get('MISSING', 0)}")
    lab = df_rec["label"].value_counts().to_dict()
    n_an = int(df_rec["overall"].isin(list(VERDICT_RANK)).sum())
    n_use = sum(lab.get(k, 0) for k in LABEL_RANK if k != "REJECT")
    out.append(f"Screening: {n_use} of {n_an} analysed recordings are usable (as they are {lab.get('USABLE', 0)}, "
               f"with flags {lab.get('USABLE_WITH_FLAGS', 0)}, after interpolation {lab.get('USABLE_NEEDS_INTERPOLATION', 0)}); "
               f"rejected {lab.get('REJECT', 0)}")
    if not df_ch.empty:
        bad_counts = df_ch[df_ch["bad"]].groupby("channel").size().sort_values(ascending=False)
        n_with_bad = int((pd.to_numeric(df_rec["n_bad_channels"], errors="coerce").fillna(0) > 0).sum())
        out.append(f"Recordings with at least one bad channel (flat, clipped, noisy or uncorrelated): {n_with_bad} of {n_an}")
        if len(bad_counts):
            out.append("Channels flagged most often: " + ", ".join(f"{k} (x{v})" for k, v in bad_counts.head(6).items()))
        else:
            out.append("No channel was flagged as flat, clipped, noisy or uncorrelated.")
        n_unc = int((pd.to_numeric(df_rec["n_uncorrelated"], errors="coerce").fillna(0) > 0).sum())
        if n_unc:
            out.append(f"Spatially uncorrelated channels (|r| < {CONFIG['corr_threshold']}) in {n_unc} recording(s)")
        n_crit = int((pd.to_numeric(df_rec["n_critical_bad"], errors="coerce").fillna(0) > 0).sum())
        if n_crit:
            names = sorted({c for s in df_rec["critical_bad_channels"] for c in str(s).split(", ") if c})
            out.append(f"Critical channels ({', '.join(CONFIG['critical_channels'])}) bad in {n_crit} recording(s): {', '.join(names)}")
    pairs = _num(df_rec[df_rec["task"] == "EyesClosed"]["alpha_ratio_EC_EO"])
    if len(pairs):
        out.append(f"Alpha blocking (Eyes Closed / Eyes Open power on O1, O2, Oz): median ratio {pairs.median():.2f}; "
                   f"{int((pairs >= CONFIG['alpha_ratio_pass']).sum())} of {len(pairs)} pairs at or above {CONFIG['alpha_ratio_pass']} "
                   "(weak alpha is only a FLAG: about 10-15% of healthy people are alpha non-responders)")
    lines = _num(df_rec["line50_db"])
    if len(lines):
        n_high = int((lines > CONFIG["line_high_db"]).sum())
        out.append(f"50 Hz line noise: median {lines.median():.1f} dB above its flanks; a notch filter is suggested for "
                   f"{int((lines > CONFIG['notch_suggested_db']).sum())} of {len(lines)} recordings (above {CONFIG['notch_suggested_db']:.0f} dB)"
                   + (f"; HIGH_LINE_NOISE (> {CONFIG['line_high_db']:.0f} dB) in {n_high}" if n_high else ""))
    return out


def headline(df_rec: pd.DataFrame, df_ch: pd.DataFrame, overall: str, label: str = "") -> str:
    """One sentence that tells a reviewer what matters most for this subject."""
    parts = [f"overall {overall}"]
    if label:
        n_use = int(df_rec["label"].isin([k for k in LABEL_RANK if k != "REJECT"]).sum())
        parts.append(f"{label}: {n_use}/{len(df_rec)} recordings usable")
    n_miss = int((df_rec["overall"] == "MISSING").sum())
    if n_miss:
        parts.append(f"{n_miss} recording(s) missing")
    n_unread = int(((df_rec["overall"] == "FAIL") & (df_rec["load_error"].astype(str) != "")).sum())
    if n_unread:
        parts.append(f"{n_unread} recording(s) unreadable")
    if not df_ch.empty:
        bad_rows = df_ch[df_ch["bad"]]
        if len(bad_rows):
            names = ", ".join(bad_rows.groupby("channel").size().sort_values(ascending=False).head(3).index)
            parts.append(f"bad channels in {bad_rows.groupby(['session', 'task', 'acq']).ngroups} recording(s) ({names})")
        else:
            parts.append("no bad channels")
    pairs = _num(df_rec[df_rec["task"] == "EyesClosed"]["alpha_ratio_EC_EO"])
    if len(pairs):
        parts.append(f"alpha blocking {'OK' if pairs.median() >= CONFIG['alpha_ratio_pass'] else 'weak (flag only)'} (median {pairs.median():.1f})")
    lines = _num(df_rec["line50_db"])
    n_notch = int((lines > CONFIG["notch_suggested_db"]).sum()) if len(lines) else 0
    if n_notch:
        parts.append(f"notch filter suggested for {n_notch}/{len(lines)} recordings")
    return "; ".join(parts)


def attention_points(df_issues: pd.DataFrame) -> list[str]:
    """Every FAIL / FLAG / MISSING finding and every PASS note as one readable line, worst first."""
    pts = []
    for r in df_issues.itertuples(index=False):
        where = f"{r.session} {r.task} {r.acq}"
        if r.code == "RECORDING_MISSING":
            pts.append(f"{where}: MISSING - recording not available")
        elif r.verdict == "PASS":
            pts.append(f"{where}: NOTE {r.code} - {r.value} ({r.limit})")
        else:
            pts.append(f"{where}: {r.verdict} {r.code} - {r.finding} = {r.value} (expected: {r.limit})")
    return pts


def write_report(rep: Report, sub: str, row: dict[str, str], elig: Eligibility, dl_info: dict[str, Any],
                 xcheck: str, df_rec: pd.DataFrame, df_sub: pd.DataFrame, df_ch: pd.DataFrame,
                 df_issues: pd.DataFrame, df_signoff: pd.DataFrame, fig_errors: list[str]) -> None:
    bar = "=" * 72
    rep.add(bar)
    rep.add(f"QC REPORT  {sub}   (OpenNeuro {DATASET_ID}, dataset version {DATASET_VERSION}; first-pass screening)")
    rep.add(bar)
    rep.add(f"Generated   : {datetime.now().astimezone().isoformat(timespec='seconds')}")
    rep.add(f"QC logic    : {QC_LOGIC_VERSION}")
    rep.add(f"Participant : sex {row['sex']}, age {row['age']}, handedness {row['handedness']}")
    rep.add(f"Sessions    : {', '.join(elig.sessions)}")
    for n in elig.notes:
        rep.add(f"Note        : {n}")
    rep.add(f"Download    : {dl_info.get('method')}" +
            (f" | repo commit {dl_info['repo_commit'][:12]}" if dl_info.get("repo_commit") else "") +
            (f" | tag {dl_info['repo_tag']}" if dl_info.get("repo_tag") else ""))
    rep.add(f"participants.tsv cross-check: {xcheck}")

    rep.add("\nSCREENING DECISION")
    rep.add(f"  Subject: verdict {df_sub.iloc[0]['overall']}, label {df_sub.iloc[0]['label']}")
    rep.add("  Labels: USABLE = nothing to note | USABLE_WITH_FLAGS = usable, flags worth a look | "
            "USABLE_NEEDS_INTERPOLATION = usable after interpolating the bad channels | REJECT = FAIL, not usable")
    rep.add("  Subject level: PARTIALLY_USABLE = some recordings are usable and some are rejected (keep the usable ones).")
    rep.add(f"  Only a FAIL is rejected. Bad channels (up to {CONFIG['interpolatable_max_channels']}, at most one critical) are repairable; "
            "weak alpha and 50 Hz line noise are FLAGS only.")

    rep.add("\nKEY FINDINGS")
    for ln in key_finding_lines(df_rec, df_ch):
        rep.add("  " + ln)

    rep.add("\nVERDICT PER RECORDING (PASS = nothing to note, FLAG = small problem, FAIL = clear problem, MISSING = file not available)")
    cols = ["session", "task", "acq", "overall", "label", "verdict_header", "verdict_integrity", "verdict_amplitude",
            "verdict_spectral", "verdict_consistency", "n_bad_channels", "n_critical_bad", "alpha_ratio_EC_EO", "line50_db"]
    rep.add(df_rec[cols].to_string(index=False))
    rep.add("\nREASON CODES PER RECORDING")
    rep.add(df_rec[["session", "task", "acq", "codes"]].to_string(index=False, max_colwidth=100))

    rep.add(f"\nFINDINGS (FAIL / MISSING / FLAG, and PASS notes with a code): {len(df_issues)}")
    if df_issues.empty:
        rep.add("  No FLAG or FAIL. All analysed recordings passed.")
    else:
        rep.add(df_issues.to_string(index=False, max_colwidth=90))

    rep.add("\nSIGN-OFF (the suggestion is only a hint; the reviewer decides)")
    rep.add(df_signoff.to_string(index=False))
    if fig_errors:
        rep.add("\nFIGURE WARNINGS")
        for e in fig_errors:
            rep.add(f"  {e}")
    rep.add("")


# ======================================================================
# MANIFEST (everything needed to reproduce / audit the run)
# ======================================================================
def _version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


def write_manifest(out_dir: Path, args: argparse.Namespace, sub: str, row: dict[str, str], elig: Eligibility,
                   dl_info: dict[str, Any], xcheck: str, data_dir: Path, repo_dir: Path, overall: str,
                   fig_errors: list[str], t_start: float) -> None:
    inputs = []
    for kind, rel in wanted_files(sub, elig.sessions):
        p = data_dir / rel
        if p.exists():
            inputs.append({"path": rel, "kind": kind, "bytes": p.stat().st_size, "sha256": _sha256(p)})
    script = Path(__file__).resolve()
    manifest = {
        "script": script.name,
        "script_sha256": _sha256(script),
        "qc_logic_version": QC_LOGIC_VERSION,
        "qc_signature": qc_signature(args.dataset_version),
        "started": datetime.fromtimestamp(t_start).astimezone().isoformat(timespec="seconds"),
        "run_seconds": round(time.time() - t_start, 1),
        "command_line": sys.argv,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "subject": sub,
        "participants_row": row,
        "eligibility": {"ok": elig.ok, "sessions_analysed": elig.sessions, "notes": elig.notes},
        "dataset": {"id": DATASET_ID, "version_requested": args.dataset_version, **read_dataset_description(repo_dir)},
        "download": dl_info,
        "participants_tsv_crosscheck": xcheck,
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     **{p: _version(p) for p in ("mne", "edfio", "numpy", "scipy", "pandas", "matplotlib", "datalad")}},
        "config": CONFIG,
        "input_files": inputs,
        "subject_overall_verdict": overall,
        "figure_warnings": fig_errors,
        "output_files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*")
                               if p.is_file() and p.name != "run_manifest.json"),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


# ======================================================================
# MAIN
# ======================================================================
class Tee:
    """Send everything that is printed to the screen AND (without colours) to run_log.txt."""

    def __init__(self, console, log) -> None:
        self.console, self.log = console, log

    def write(self, s: str) -> int:
        self.console.write(s)
        self.log.write(_ANSI_RE.sub("", s))
        return len(s)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name: str):
        return getattr(self.console, name)


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def fmt_selection(subjects: list[str]) -> str:
    """['sub-001','sub-002','sub-003','sub-007'] -> '1-3, 7'"""
    nums = sorted(int(s[4:]) for s in subjects)
    runs: list[str] = []
    start = prev = None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            runs.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = n
    if start is not None:
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(runs)


# ======================================================================
# EARLIER RESULTS (cache) AND CLEAN-UP
# ======================================================================
def load_cached_status(subj_dir: Path, signature: str) -> tuple[dict[str, Any] | None, str | None]:
    """Were this subject's results already produced with the SAME QC logic and thresholds?

    Returns (status dict, None) if yes (nothing needs to be downloaded or analysed), otherwise (None, reason or None).
    """
    p = subj_dir / "status.json"
    if not p.exists():
        return None, None
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, "an unreadable status.json was found: the subject is analysed again"
    if s.get("status") != "COMPLETED":
        return None, None
    needed = ["qc_results/df_recordings.csv", "qc_results/qc_report.txt", "qc_results/run_manifest.json"]
    if any(not (subj_dir / n).exists() for n in needed):
        return None, "earlier results are incomplete (files are missing): the subject is analysed again"
    if s.get("qc_signature") != signature:
        return None, "earlier results come from a different QC version or other thresholds: the subject is analysed again"
    return s, None


def cleanup_downloaded(sub: str, sessions: list[str], data_dir: Path, repo_dir: Path) -> dict[str, Any]:
    """Free disk space after a successful analysis.

    1. deletes the subject's EDF / json / events.tsv copies in the data folder (only those files, nothing else);
    2. drops the downloaded content from the DataLad clone ('datalad drop'): it can be fetched again from OpenNeuro.
    """
    info: dict[str, Any] = {"performed": True, "reason": "", "data_files_deleted": 0, "annex_files_dropped": 0,
                            "bytes_freed": 0, "errors": []}
    wanted = wanted_files(sub, sessions)
    for _kind, rel in wanted:
        p = data_dir / rel
        if p.is_file():
            try:
                size = p.stat().st_size
                p.unlink()
                info["bytes_freed"] += size
                info["data_files_deleted"] += 1
            except OSError as e:
                info["errors"].append(f"could not delete {p}: {e}")
    sub_dir = data_dir / sub
    if sub_dir.exists():                                   # remove folders that are now empty
        for d in sorted((q for q in sub_dir.rglob("*") if q.is_dir()), key=lambda q: len(q.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
        try:
            sub_dir.rmdir()
        except OSError:
            pass
    if (repo_dir / ".git").exists() and shutil.which("datalad"):
        edfs = [rel for k, rel in wanted if k == "edf" and (repo_dir / rel).exists()]    # exists = content is present
        if edfs:
            sizes = {rel: (repo_dir / rel).stat().st_size for rel in edfs}
            rc, tail = _run_quiet(["datalad", "drop", *edfs], cwd=repo_dir)
            left = [rel for rel in edfs if (repo_dir / rel).exists()]
            info["annex_files_dropped"] = len(edfs) - len(left)
            info["bytes_freed"] += sum(sz for rel, sz in sizes.items() if rel not in left)
            if left:
                info["errors"].append(f"'datalad drop' could not remove {len(left)} file(s) (exit code {rc}): {tail.splitlines()[-1] if tail else ''}")
    return info


# ======================================================================
# STATUS FILES (one per subject, also for rejected subjects)
# ======================================================================
def render_status_text(s: dict[str, Any]) -> str:
    w = 76
    L: list[str] = ["=" * w]
    L.append(f"STATUS  {s['subject']}     {s['status']}" + (f"     QC verdict: {s['qc_verdict']}" if s.get("qc_verdict") else "")
             + (f"     {s['qc_label']}" if s.get("qc_label") else ""))
    L.append("=" * w)
    L += textwrap.wrap(s["headline"], w, initial_indent="Summary     : ", subsequent_indent=" " * 14)
    L.append(f"Started     : {s['started']}")
    L.append(f"Finished    : {s['finished']}   (duration {s['duration_s']} s)")
    p = s["participant"]
    L.append(f"Participant : sex {p['sex']}, age {p['age']}, handedness {p['handedness']}")
    if s.get("qc_logic_version"):
        L.append(f"QC logic    : {s['qc_logic_version']}   (fingerprint {s.get('qc_signature')})")

    e = s["eligibility"]
    L += ["", "ELIGIBILITY  ->  " + ("eligible" if e["eligible"] else "NOT ELIGIBLE (rejected)")]
    for r in e["rules"]:
        L.append(f"  [{r['result']:<4}] {r['rule']:<22} {r['detail']}")
    if not e["eligible"]:
        L.append("  Reason(s) for rejection:")
        L += [f"    - {x}" for x in e["reasons"]]
        L.append("  Nothing was downloaded and no QC was run for this subject.")
    else:
        L.append(f"  Sessions analysed: {', '.join(e['sessions_to_analyse'])}")
        L += [f"  Note: {n}" for n in e["notes"]]

    d = s.get("download")
    if d:
        L += ["", "DATA"]
        tag = f" | dataset tag {d['repo_tag']}" if d.get("repo_tag") else ""
        commit = f" | commit {d['repo_commit'][:12]}" if d.get("repo_commit") else ""
        L.append(f"  Download method : {d.get('method')}{tag}{commit}")
        L.append(f"  Files           : downloaded {len(d.get('downloaded', []))} | already on disk "
                 f"{len(d.get('already_present', []))} | not in the dataset {len(d.get('not_in_dataset', []))} "
                 f"| failed {len(d.get('failed', []))}")
        absent = [Path(x).name for x in d.get("not_in_dataset", []) if x.endswith("_eeg.edf")]
        if absent:
            L.append("  EDF not in the dataset: " + ", ".join(absent))
        failed = [Path(x).name for x in d.get("failed", [])]
        if failed:
            L.append("  Download failed for    : " + ", ".join(failed))
        if d.get("error"):
            L.append(f"  Download error  : {d['error']}")
        if d.get("participants_tsv_crosscheck"):
            L.append(f"  participants.tsv cross-check: {d['participants_tsv_crosscheck']}")

    if s.get("recordings"):
        c = s["counts"]
        L += ["", "QC RESULT (first-pass screening)",
              f"  Subject verdict : {s['qc_verdict'] or 'NO DATA'}   screening label: {s.get('qc_label') or '-'}",
              f"  Recordings      : expected {c.get('expected')} | PASS {c.get('PASS', 0)}, FLAG {c.get('FLAG', 0)}, "
              f"FAIL {c.get('FAIL', 0)}, MISSING {c.get('MISSING', 0)}",
              f"  Screening       : usable as is {c.get('USABLE', 0)} | usable with flags {c.get('USABLE_WITH_FLAGS', 0)} | "
              f"needs interpolation {c.get('USABLE_NEEDS_INTERPOLATION', 0)} | rejected {c.get('REJECT', 0)}", ""]
        tbl = pd.DataFrame(s["recordings"]).rename(columns={"overall": "verdict", "n_bad_channels": "n_bad",
                                                            "alpha_ratio_EC_EO": "alpha_EC/EO", "line50_db": "50Hz_dB"})
        tbl["n_bad"] = tbl["n_bad"].map(lambda v: "-" if pd.isna(v) else int(v))
        tbl = tbl.astype(object).where(tbl.notna(), "-").replace("", "-")
        shown = ["session", "task", "acq", "verdict", "label", "n_bad", "bad_channels", "alpha_EC/EO", "50Hz_dB"]
        L += ["  " + ln for ln in tbl[shown].to_string(index=False).splitlines()]
        L += ["", "  Reason codes:"]
        for r in s["recordings"]:
            L.append(f"   - {r['session']} {r['task']} {r['acq']}: {r.get('codes') or '-'}")
        if s.get("key_findings"):
            L += ["", "  Key findings:"] + [f"   - {x}" for x in s["key_findings"]]
        L += ["", "  Needs attention:"]
        L += ([f"   - {x}" for x in s["attention"]] if s.get("attention") else ["   - nothing: no FLAG, FAIL or MISSING"])

    cl = s.get("cleanup")
    if cl:
        L += ["", "CLEAN-UP"]
        if cl.get("performed"):
            L.append(f"  Analysed data deleted: {cl.get('data_files_deleted', 0)} file(s) in the data folder, "
                     f"{cl.get('annex_files_dropped', 0)} DataLad file(s) dropped, {cl.get('bytes_freed', 0) / 1e6:.0f} MB freed")
            L += [f"  Problem: {x}" for x in cl.get("errors", [])]
        else:
            L.append(f"  Data kept: {cl.get('reason')}")

    if s.get("files"):
        L += ["", "FILES (relative to this folder)"] + [f"  {x}" for x in s["files"]]
    if s.get("warnings"):
        L += ["", "WARNINGS"] + [f"  - {x}" for x in s["warnings"]]
    if s.get("error"):
        L += ["", "ERROR", f"  {s['error']['type']}: {s['error']['message']}"] + [f"  | {x}" for x in s["error"]["traceback"]]
    return "\n".join(L) + "\n"


def write_status(subj_dir: Path, s: dict[str, Any]) -> None:
    (subj_dir / "status.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    (subj_dir / "status.txt").write_text(render_status_text(s), encoding="utf-8")


# ======================================================================
# CONSOLE BLOCKS
# ======================================================================
def padv(value: str, width: int) -> str:
    return vc(value) + " " * max(0, width - len(value))


def subject_header(idx: int, total: int, sub: str, row: dict[str, str], t_batch: float) -> None:
    hand = {"right": "right-handed", "left": "left-handed"}.get(row["handedness"], f"handedness {row['handedness']}")
    print()
    rule("=")
    say(f"{paint(f'[{idx}/{total}]', 'cyan', 'bold')}  {paint(sub, 'bold')}     sex {row['sex']}, age {row['age']}, {hand}"
        f"     {paint('elapsed ' + fmt_duration(time.time() - t_batch), 'dim')}", 1)
    rule("=")


def print_eligibility(elig: Eligibility) -> None:
    for r in elig.rules:
        tag = paint(f"{r['result']:<4}", STATUS_COLOR[r["result"]], "bold")
        say(f"[{tag}] {r['rule']:<22} {r['detail']}", 4)
    if elig.ok:
        good(f"Eligible. Sessions to analyse: {', '.join(elig.sessions)}")
    else:
        bad(f"NOT eligible: {'; '.join(elig.reasons)}")


def print_result(s: dict[str, Any], status_path: Path) -> None:
    print()
    rule("-")
    qc = f"   QC verdict: {vc(s['qc_verdict'])}" if s.get("qc_verdict") else ""
    lab = f"   {vc(s['qc_label'])}" if s.get("qc_label") else ""
    reused = paint("   (reused earlier results)", "dim") if s.get("reused") else ""
    say(f"{paint('RESULT', 'bold')}   {vc(s['status'])}{qc}{lab}{reused}", 1)
    for ln in textwrap.wrap(s["headline"], term_width() - 12) or [""]:
        say(ln, 10)
    say(paint(f"status file: {status_path}", "dim"), 10)
    rule("-")


# ======================================================================
# ONE SUBJECT
# ======================================================================
N_STEPS = 5


def process_subject(sub: str, idx: int, total: int, args: argparse.Namespace,
                    results: list[dict[str, Any]], t_batch: float) -> None:
    """Eligibility -> (reuse earlier results?) -> download -> QC -> tables, figures, report -> clean-up.
    Always writes <out-root>/<sub>/status.txt|json (except when earlier results are reused untouched).

    A problem in one subject never stops the others (except Ctrl+C).
    """
    t_start = time.time()
    row = PARTICIPANTS[sub]
    data_dir, repo_dir = args.data_dir, args.repo_dir
    subj_dir = args.out_root / sub
    qc_dir = subj_dir / "qc_results"
    fig_dir = qc_dir / "figures"
    elig = check_eligibility(row)
    signature = qc_signature(args.dataset_version)

    # ---------------- results of an earlier run with the same QC logic: nothing to do ----------------
    why_again = None
    if elig.ok and not args.force_reanalysis:
        cached, why_again = load_cached_status(subj_dir, signature)
        if cached is not None:
            subject_header(idx, total, sub, row, t_batch)
            step(1, N_STEPS, "Eligibility check")
            print_eligibility(elig)
            step(2, N_STEPS, "Earlier results")
            good("This subject was already analysed with the same QC logic and thresholds.")
            say(f"Analysed on {cached.get('finished')}: verdict {cached.get('qc_verdict')}, {cached.get('qc_label')}", 4)
            note("Nothing is downloaded and nothing is analysed again (use --force-reanalysis to run it again).")
            cached["reused"] = True
            print_result(cached, subj_dir / "status.txt")
            results.append(cached)
            return

    status: dict[str, Any] = {
        "subject": sub, "status": "STARTED", "qc_verdict": None, "qc_label": None, "headline": "",
        "qc_logic_version": QC_LOGIC_VERSION, "qc_signature": signature,
        "started": datetime.fromtimestamp(t_start).astimezone().isoformat(timespec="seconds"),
        "finished": None, "duration_s": None,
        "participant": {"sex": row["sex"], "age": row["age"], "handedness": row["handedness"]},
        "eligibility": {"eligible": elig.ok, "rules": elig.rules, "reasons": elig.reasons, "notes": elig.notes,
                        "sessions_to_analyse": elig.sessions if elig.ok else []},
        "download": None, "cleanup": None, "recordings": [], "counts": {}, "key_findings": [], "attention": [],
        "figures": [], "warnings": [], "files": [], "error": None, "reused": False,
        "folders": {"subject": str(subj_dir), "qc_results": str(qc_dir) if elig.ok else None},
    }
    subj_dir.mkdir(parents=True, exist_ok=True)
    real_stdout, log_fh = sys.stdout, None
    try:
        if elig.ok:
            fig_dir.mkdir(parents=True, exist_ok=True)
            for p_ in fig_dir.glob("*.png"):              # no stale files from an earlier run
                p_.unlink()
            for name in OUTPUT_FILES:
                (qc_dir / name).unlink(missing_ok=True)
            log_fh = open(qc_dir / "run_log.txt", "w", encoding="utf-8")
            sys.stdout = Tee(real_stdout, log_fh)

        subject_header(idx, total, sub, row, t_batch)

        # ---------------- step 1: eligibility (before anything is downloaded) ----------------
        step(1, N_STEPS, "Eligibility check")
        print_eligibility(elig)
        if not elig.ok:
            status.update(status="REJECTED", headline="Rejected: " + "; ".join(elig.reasons))
            note("Nothing is downloaded and no QC is run. The reason is saved in this subject's status file.")
            return
        for n in elig.notes:
            note(n)
        if why_again:
            note(why_again[0].upper() + why_again[1:] + ".")

        # ---------------- step 2: data ----------------
        step(2, N_STEPS, "Data: download")
        dl_info: dict[str, Any] = {"method": "skipped (--no-download)", "downloaded": [], "already_present": [],
                                   "not_in_dataset": [], "failed": []}
        if args.no_download:
            note(f"Download skipped (--no-download). Using the files in {data_dir}")
        else:
            try:
                dl_info = fetch_data(sub, elig.sessions, data_dir, repo_dir, args.download_method,
                                     args.dataset_version, args.force_download)
            except Exception as e:
                if args.debug:
                    raise
                bad(f"Download problem: {type(e).__name__}: {e}")
                note("Continuing with the files that are already in the data folder.")
                dl_info.update(method=args.download_method, error=f"{type(e).__name__}: {e}")

        xcheck = "participants.tsv not found (check skipped)"
        for cand in (repo_dir / "participants.tsv", data_dir / "participants.tsv"):
            if cand.exists():
                try:
                    n, diffs = compare_with_tsv(cand, only_sub=sub)
                    if n == 0:
                        xcheck = f"{sub} not found in {cand}"
                    elif diffs:
                        xcheck = "DIFFERENCES to " + str(cand) + ": " + "; ".join(diffs)
                    else:
                        xcheck = f"OK, the embedded row is identical to {cand}"
                except Exception as e:
                    xcheck = f"could not read {cand}: {type(e).__name__}: {e}"
                break
        if xcheck.startswith("OK"):
            good("participants.tsv cross-check: OK (age, sessions and late triggers match the dataset)")
        elif xcheck.startswith("DIFFERENCES"):
            warn(f"participants.tsv cross-check: {xcheck}")
            warn("The embedded table differs from the dataset's participants.tsv. Check the eligibility decision above.")
        else:
            note(f"participants.tsv cross-check: {xcheck}")
        dl_info["participants_tsv_crosscheck"] = xcheck
        status["download"] = dl_info

        # ---------------- step 3: QC ----------------
        step(3, N_STEPS, "Quality-control analysis")
        st, _inventory = run_qc(data_dir, sub, elig.sessions, debug=args.debug)
        df_full, df_rec, df_sub, df_ch = build_tables(st, sub, row, elig.sessions)
        df_issues = build_issues(st)
        overall = str(df_sub.iloc[0]["overall"])
        subject_label = str(df_sub.iloc[0]["label"])
        say(paint("Verdict per recording", "bold"), 4)
        short = {"EyesClosed": "EC", "EyesOpen": "EO"}
        for r in df_rec.itertuples(index=False):
            bits: list[str] = []
            if r.overall == "MISSING":
                bits.append("file not available")
            elif r.load_error:
                bits.append("file could not be read")
            else:
                if pd.notna(r.n_bad_channels) and int(r.n_bad_channels) > 0:
                    bits.append(f"bad: {r.bad_detail}")
                if r.task == "EyesClosed" and pd.notna(r.alpha_ratio_EC_EO):
                    bits.append(f"alpha EC/EO {r.alpha_ratio_EC_EO:.1f}")
                if pd.notna(r.line50_db):
                    bits.append(f"50 Hz {r.line50_db:.0f} dB")
            say(f"{r.session}  {short.get(r.task, r.task)}  {r.acq:<4}  {padv(r.overall, 8)}{padv(r.label, 28)}{' | '.join(bits)}", 6)

        # ---------------- step 4: tables, figures, report ----------------
        step(4, N_STEPS, "Saving tables, figures and report")
        suggest = {"USABLE": "accept", "USABLE_WITH_FLAGS": "accept with note",
                   "USABLE_NEEDS_INTERPOLATION": "accept (interpolate the bad channels in preprocessing)",
                   "PARTIALLY_USABLE": "accept the usable recordings; re-check the rejected ones",
                   "REJECT": "report / re-check", "NO DATA": "no recording could be analysed"}
        df_signoff = pd.DataFrame({
            "subject": df_sub["subject"], "overall_verdict": df_sub["overall"], "screening_label": df_sub["label"],
            "suggested_decision": df_sub["label"].map(suggest),
            "reviewer": "", "date": datetime.now().date().isoformat(),
            "decision (accept / accept with note / re-check)": "",
        })
        with Task("Saving tables (CSV)") as t:
            df_rec.to_csv(qc_dir / "df_recordings.csv", index=False)
            df_full.to_csv(qc_dir / "df_recordings_full.csv", index=False)
            df_sub.to_csv(qc_dir / "df_subjects.csv", index=False)
            df_ch.to_csv(qc_dir / "df_channels.csv", index=False)
            df_issues.to_csv(qc_dir / "qc_issues.csv", index=False)
            df_signoff.to_csv(qc_dir / "qc_signoff.csv", index=False)
            pivot = df_rec.pivot_table(index="subject", columns=["session", "task", "acq"], values="overall", aggfunc="first")
            pivot = pivot[sorted(pivot.columns, key=lambda c: (c[0], TASKS.index(c[1]), ACQS.index(c[2])))]
            pivot.to_csv(qc_dir / "qc_pivot.csv")
            t.detail = "7 files"

        fig_res: dict[str, list[str]] = {"saved": [], "errors": [], "notes": []}
        if df_ch.empty:
            warn("No recording could be read, so there is nothing to plot.")
        else:
            with Task("Drawing figures (A-G and the channel quality topomap)") as t:
                fig_res = make_figures(st, sub, df_rec, df_ch, fig_dir)
                t.detail = f"{len(fig_res['saved'])} figures saved" + (f", {len(fig_res['errors'])} failed" if fig_res["errors"] else "")
            for e_ in fig_res["errors"]:
                warn(e_, 8)
            for n_ in fig_res["notes"]:
                note(n_, 8)

        with Task("Writing the report and the run manifest") as t:
            rep = Report()
            write_report(rep, sub, row, elig, dl_info, xcheck, df_rec, df_sub, df_ch, df_issues, df_signoff, fig_res["errors"])
            rep.save(qc_dir / "qc_report.txt")
            write_manifest(qc_dir, args, sub, row, elig, dl_info, xcheck, data_dir, repo_dir, overall, fig_res["errors"], t_start)

        counts = {"expected": len(df_rec),
                  **{k: int(df_sub.iloc[0][f"n_{k}"]) for k in ("PASS", "FLAG", "FAIL", "MISSING", "USABLE", "USABLE_WITH_FLAGS",
                                                               "USABLE_NEEDS_INTERPOLATION", "REJECT")}}
        cols = ["session", "task", "acq", "overall", "label", "codes", "n_bad_channels", "bad_channels", "bad_detail",
                "n_critical_bad", "n_uncorrelated", "frozen_s", "alpha_ratio_EC_EO", "line50_db", "notch_suggested"]
        no_data = overall == "NO DATA"
        status.update(
            status="NO_DATA" if no_data else "COMPLETED",
            qc_verdict=None if no_data else overall,
            qc_label=None if no_data else subject_label,
            headline=(headline(df_rec, df_ch, overall, subject_label) if not no_data else
                      f"No recording could be analysed: all {len(df_rec)} expected recordings are missing or unreadable"),
            recordings=json.loads(df_rec[cols].to_json(orient="records")),
            counts=counts, key_findings=key_finding_lines(df_rec, df_ch), attention=attention_points(df_issues),
            figures=fig_res["saved"], warnings=fig_res["errors"] + fig_res["notes"])

        # ---------------- step 5: clean-up (the analysed data are no longer needed) ----------------
        step(5, N_STEPS, "Clean-up")
        if no_data:
            status["cleanup"] = {"performed": False, "reason": "no recording could be analysed"}
            note("Nothing to clean up.")
        elif args.no_download:
            status["cleanup"] = {"performed": False, "reason": "--no-download: your own data files are never deleted"}
            note("Data files kept (--no-download: your own files are never deleted).")
        elif args.keep_data:
            status["cleanup"] = {"performed": False, "reason": "--keep-data"}
            note("Data files kept (--keep-data).")
        else:
            with Task("Deleting the analysed data (data folder and DataLad copy)") as t:
                status["cleanup"] = cleanup_downloaded(sub, elig.sessions, data_dir, repo_dir)
                cl = status["cleanup"]
                t.detail = (f"{cl['data_files_deleted']} data file(s) deleted, {cl['annex_files_dropped']} DataLad file(s) "
                            f"dropped, {cl['bytes_freed'] / 1e6:.0f} MB freed")
            for err in status["cleanup"]["errors"]:
                warn(err, 8)

    except KeyboardInterrupt:
        status.update(status="INTERRUPTED", headline="Stopped by the user (Ctrl+C) before this subject was finished")
        raise
    except Exception as e:
        if args.debug:
            raise
        status.update(status="ERROR", headline=f"Unexpected error: {type(e).__name__}: {e}",
                      error={"type": type(e).__name__, "message": str(e),
                             "traceback": traceback.format_exc().strip().splitlines()[-12:]})
        bad(f"Unexpected error: {type(e).__name__}: {e}")
        note("The other subjects continue. The data files are kept. Details are in this subject's status file.")
    finally:
        if status["status"] == "STARTED":
            status["status"] = "ERROR"
            status["headline"] = status["headline"] or "The run ended unexpectedly"
        status["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        status["duration_s"] = round(time.time() - t_start, 1)
        if qc_dir.exists():
            status["files"] = sorted(str(p.relative_to(subj_dir)) for p in qc_dir.rglob("*")
                                     if p.is_file() and p.parent.name != "figures")
            status["files"] += sorted(str(p.relative_to(subj_dir)) for p in fig_dir.glob("*.png")) if fig_dir.exists() else []
        status["files"] = ["status.txt", "status.json"] + status["files"]
        try:
            write_status(subj_dir, status)
        except Exception as werr:                  # never hide the real problem behind a status-writing problem
            print(f"WARNING: could not write the status file: {werr}", file=sys.stderr)
        print_result(status, subj_dir / "status.txt")
        results.append(status)
        sys.stdout = real_stdout
        if log_fh is not None:
            log_fh.close()


# ======================================================================
# BATCH SUMMARY
# ======================================================================
def update_overview_csv(out_root: Path, results: list[dict[str, Any]]) -> Path:
    """summary_all_subjects.csv collects ALL subjects processed so far (also from earlier runs)."""
    rows = [{"subject": s["subject"], "status": s["status"], "qc_verdict": s["qc_verdict"] or "", "screening_label": s.get("qc_label") or "",
             "headline": s["headline"], "sex": s["participant"]["sex"], "age": s["participant"]["age"],
             "recordings_expected": s["counts"].get("expected", ""), "PASS": s["counts"].get("PASS", ""),
             "FLAG": s["counts"].get("FLAG", ""), "FAIL": s["counts"].get("FAIL", ""),
             "MISSING": s["counts"].get("MISSING", ""),
             "usable_recordings": (sum(s["counts"].get(k, 0) for k in ("USABLE", "USABLE_WITH_FLAGS", "USABLE_NEEDS_INTERPOLATION"))
                                   if s["counts"] else ""),
             "reused": bool(s.get("reused")), "finished": s["finished"], "folder": s["folders"]["subject"]}
            for s in results]
    new = pd.DataFrame(rows).astype(str)
    path = out_root / "summary_all_subjects.csv"
    df = new
    if path.exists():
        try:
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            df = pd.concat([old[~old["subject"].isin(new["subject"])], new], ignore_index=True)
        except Exception:
            df = new
    df.sort_values("subject").to_csv(path, index=False)
    return path


def print_batch_summary(results: list[dict[str, Any]], elapsed: float) -> None:
    print()
    rule("=")
    say(f"{paint('SUMMARY', 'bold')}   {len(results)} subject(s) in {fmt_duration(elapsed)}", 1)
    rule("=")
    say(paint(f"{'Subject':<9}{'Status':<13}{'QC':<7}{'Screening label':<29}Details", "dim"), 1)
    limit = max(20, term_width() - 60)
    for s in results:
        qc = s["qc_verdict"] or "-"
        lab = s.get("qc_label") or "-"
        text = ("[reused] " if s.get("reused") else "") + s["headline"]
        detail = text if len(text) <= limit else text[:limit - 3] + "..."
        say(f"{s['subject']:<9}{padv(s['status'], 13)}{padv(qc, 7)}{padv(lab, 29)}{detail}", 1)
    by = Counter(s["status"] for s in results)
    qc_counts = Counter(s["qc_verdict"] for s in results if s["qc_verdict"])
    lab_counts = Counter(s["qc_label"] for s in results if s.get("qc_label"))
    n_reused = sum(1 for s in results if s.get("reused"))
    print()
    say(f"Completed {by.get('COMPLETED', 0)}"
        + (f"  ({', '.join(f'{k} {v}' for k, v in sorted(qc_counts.items()))})" if qc_counts else "")
        + (f", {n_reused} reused from earlier runs" if n_reused else "")
        + f"   |   rejected {by.get('REJECTED', 0)}   |   no data {by.get('NO_DATA', 0)}   |   errors {by.get('ERROR', 0)}"
        + (f"   |   interrupted {by.get('INTERRUPTED', 0)}" if by.get("INTERRUPTED") else ""), 1)
    if lab_counts:
        say("Screening: " + "   ".join(f"{k} {lab_counts[k]}" for k in SUBJECT_LABELS if lab_counts.get(k)), 1)


# ======================================================================
# MAIN
# ======================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="qc_eeg_subject.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="First-pass screening (triage) of the resting-state EEG of one or many subjects of OpenNeuro ds005385.",
        epilog=("Examples:\n"
                "  python qc_eeg_subject.py              asks which subjects\n"
                "  python qc_eeg_subject.py 18           one subject\n"
                "  python qc_eeg_subject.py 1 2 3        several subjects\n"
                "  python qc_eeg_subject.py 1 to 20      a range\n"
                "  python qc_eeg_subject.py 1 5 to 8 12  mixed\n\n"
                "Exit codes: 0 = everything finished, 1 = a technical problem (error / no data / Ctrl+C),\n"
                "            2 = invalid input, or every requested subject was rejected."))
    p.add_argument("subjects", nargs="*", help=f"subject numbers 1-{N_PARTICIPANTS}: '1 2 3' or '1 to 20' (no 'sub-' needed). If omitted you are asked.")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="where the EDF files are copied to / read from (default: data)")
    p.add_argument("--repo-dir", type=Path, default=Path(f"{DATASET_ID}_repo"), help=f"DataLad clone of the dataset (default: {DATASET_ID}_repo)")
    p.add_argument("--out-root", type=Path, default=Path("qc_output"),
                   help="results folder: <out-root>/<sub-XXX>/status.txt and <out-root>/<sub-XXX>/qc_results/ (default: qc_output)")
    p.add_argument("--dataset-version", default=DATASET_VERSION, help=f"dataset version tag (default: {DATASET_VERSION})")
    p.add_argument("--download-method", choices=["auto", "datalad", "https"], default="auto",
                   help="auto = DataLad if available, otherwise plain HTTPS")
    p.add_argument("--no-download", action="store_true",
                   help="do not download anything; use the files already in --data-dir (they are never deleted)")
    p.add_argument("--force-download", action="store_true", help="download again even when the files already exist")
    p.add_argument("--force-reanalysis", action="store_true",
                   help="analyse again even when results of the same QC version already exist for the subject")
    p.add_argument("--keep-data", action="store_true",
                   help="do not delete the downloaded data after the analysis (default: they are deleted to save disk space)")
    p.add_argument("--verify-table", metavar="TSV", help="compare the embedded participant table with a TSV file and exit")
    p.add_argument("--debug", action="store_true", help="show the full traceback and stop, instead of recording the problem in status.txt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_table:
        return verify_table(args.verify_table)

    if args.subjects:
        subjects, errors = parse_selection(args.subjects)
        if errors or not subjects:
            for e in errors or ["no subject numbers found"]:
                print(f"ERROR: {e}", file=sys.stderr)
            print("Examples:   1 2 3      |      1 to 20      |      1 5 to 8 12", file=sys.stderr)
            return 2
    else:
        subjects = ask_subjects()

    sig = qc_signature(args.dataset_version)
    n_ok = n_reuse = 0
    for s in subjects:
        if check_eligibility(PARTICIPANTS[s]).ok:
            n_ok += 1
            if not args.force_reanalysis and load_cached_status(args.out_root / s, sig)[0] is not None:
                n_reuse += 1
    print()
    rule("=")
    say(f"{paint('EEG QC', 'bold')}  |  first-pass screening  |  OpenNeuro {DATASET_ID} v{args.dataset_version}", 1)
    say(f"Subjects : {len(subjects)}  ({fmt_selection(subjects)})", 1)
    what = ("files already in the data folder, kept" if args.no_download
            else "downloaded, checked, kept" if args.keep_data else "downloaded, checked, then deleted")
    say(f"Plan     : {n_ok - n_reuse} to analyse ({what}), {n_reuse} already analysed (reused), "
        f"{len(subjects) - n_ok} rejected (age / late triggers: nothing is downloaded, the reason goes into the status file)", 1)
    say(f"Folders  : data = {args.data_dir}   |   results = {args.out_root}", 1)
    rule("=")
    args.out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    t_batch = time.time()
    try:
        for i, sub in enumerate(subjects, 1):
            process_subject(sub, i, len(subjects), args, results, t_batch)
    except KeyboardInterrupt:
        print()
        warn("Stopped by the user (Ctrl+C). The subjects that were finished are saved.", 1)

    if results:
        print_batch_summary(results, time.time() - t_batch)
        overview = update_overview_csv(args.out_root, results)
        print()
        say(f"Results folder : {args.out_root.resolve()}", 1)
        say(f"Overview (all subjects processed so far): {overview}", 1)
    by = Counter(s["status"] for s in results)
    if any(by.get(k) for k in ("ERROR", "NO_DATA", "INTERRUPTED")) or len(results) < len(subjects):
        return 1
    if results and by.get("REJECTED", 0) == len(results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
