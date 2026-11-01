"""Configuration: every threshold, every standard channel set, every derived rule.

Nothing in the QC engine is hard-coded. All numbers live in :class:`QCConfig`, can be overridden from the
command line (``--set KEY=VALUE``, ``--config FILE.json`` or a dedicated flag) and are stored, together with
a fingerprint, in every result so that results are only reused when the logic and the thresholds match.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

#: Version of the QC logic. Bump it whenever the meaning of a verdict changes.
QC_LOGIC_VERSION = "triage-1.0"

# ---------------------------------------------------------------------------------------------- file formats
#: EEG file formats that can be screened (BIDS ``_eeg.<ext>`` main files).
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".edf", ".bdf", ".vhdr", ".set", ".fif")

#: Companion extensions that must be fetched together with the main file.
COMPANION_EXTENSIONS: dict[str, tuple[str, ...]] = {
    ".vhdr": (".vmrk", ".eeg", ".dat"),
    ".set": (".fdt",),
}

# ---------------------------------------------------------------------------------------------- standard montage
#: Legacy 10-20 names mapped to their modern 10-10 equivalents.
LEGACY_NAME_MAP: dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

#: The classic 19-channel 10-20 system (modern names). Oz is NOT part of it.
CLASSIC_1020: tuple[str, ...] = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
                                 "P7", "P3", "Pz", "P4", "P8", "O1", "O2")

#: Core anchors used by the standard-montage gatekeeper.
CORE_ANCHORS: tuple[str, ...] = ("Fz", "Cz", "Pz", "Oz", "F3", "F4", "C3", "C4", "O1", "O2")

DEFAULT_CRITICAL_CHANNELS: tuple[str, ...] = ("Cz", "Fz", "Pz", "O1", "O2")

#: Names of non-EEG channels (matched on the whole, normalised name).
AUX_NAME_PATTERN = r"^(eeg[\s_\-]*)?(v|h)?(eog|ecg|ekg|emg)\d*$|^(trig|trigger|sti|stim|status|event|events|gsr|resp|ppg)\b"

#: ``type`` values of channels.tsv that are not scalp EEG.
AUX_TSV_TYPES: frozenset[str] = frozenset({"EOG", "VEOG", "HEOG", "ECG", "EKG", "EMG", "TRIG", "MISC", "REF",
                                           "GSR", "RESP", "PPG", "SYSCLOCK", "ADC", "DAC", "STIM"})

#: Order of the verdicts (higher = worse).
VERDICT_RANK = {"PASS": 0, "FLAG": 1, "FAIL": 2}
LABEL_RANK = {"USABLE": 0, "USABLE_WITH_FLAGS": 1, "USABLE_NEEDS_INTERPOLATION": 2, "REJECT": 3}

#: Weights of the 0-100 recording quality score (used only for ranking, never for the verdict).
SCORE_WEIGHTS: dict[str, float] = {
    "bad_channels": 25.0,
    "critical_channels": 20.0,
    "uncorrelated_channels": 10.0,
    "spatial_correlation": 15.0,
    "line_noise": 15.0,
    "frozen_segments": 10.0,
    "duration": 5.0,
}


class ConfigError(ValueError):
    """Raised for an invalid configuration value."""


def _tuplify(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_tuplify(v) for v in value)
    return value


@dataclass
class QCConfig:
    """All thresholds of the 5-layer screening. Every field has a comment saying what it does."""

    # ---- Layer 1: header and metadata ---------------------------------------------------------------
    min_eeg_channels: int = 16                    # fewer EEG channels -> FAIL (a classic 10-20 cap has 19)
    min_sfreq_hz: float = 128.0                   # lower sampling rate -> FAIL
    max_sfreq_hz: float | None = None             # higher sampling rate -> FLAG (None = no upper bound)
    duration_fail_below_s: float = 60.0           # shorter recording -> FAIL
    duration_ok_range_s: tuple[float, float] = (120.0, 3600.0)   # outside -> FLAG
    required_channels: tuple[str, ...] = ("O1", "O2")             # must exist (Oz is optional)
    min_file_kb: float = 5.0                      # smaller main file = broken download / dangling link
    montage_min_standard_fraction: float = 0.70   # share of EEG channels with a standard 10-20/10-10/10-05 name
    montage_min_anchors: int = 6                  # how many of the 10 core anchors must be present
    non_eeg_name_hints: tuple[str, ...] = ("EOG", "VEOG", "HEOG", "ECG", "EKG", "EMG", "TRIG", "STI", "STATUS")

    # ---- Layer 2: channel integrity -----------------------------------------------------------------
    detrend_method: str = "highpass"              # "highpass", "linear" or "none" (baseline correction, RAM only)
    detrend_hp_hz: float = 0.5
    detrend_hp_order: int = 2
    flat_std_uv: float = 0.1                      # raw STD below this = flat channel
    clip_fraction: float = 0.001                  # share of samples exactly at the channel min/max = clipped channel
    frozen_min_s: float = 0.1                     # all channels unchanged for this long = frozen segment (0.1 s = 100 samples at 1 kHz)
    noisy_ratio: float = 4.0                      # typical amplitude above ratio * median of all channels = noisy
    noisy_min_uv: float = 50.0                    # ... and above this absolute value
    global_frozen_flag_s: float = 5.0             # frozen time above this -> FLAG
    global_frozen_fail_fraction: float = 0.10     # frozen share above this -> FAIL
    corr_threshold: float = 0.20                  # |r| with the median of all channels below this = uncorrelated
    corr_threshold_relaxed: float = 0.18          # same, for the occipital channels (strong local alpha)
    corr_relaxed_channels: tuple[str, ...] = ("O1", "O2", "Oz")
    corr_band_hz: tuple[float, float] = (1.0, 45.0)
    corr_window_s: float = 5.0
    corr_target_fs_hz: float = 125.0
    corr_unreliable_fraction: float = 0.50        # more than this share below threshold: reference not trusted
    corr_scale_by_channel_count: bool = True      # relax the correlation thresholds on caps with fewer channels
    corr_reference_n_channels: int = 64           # the channel count the thresholds were designed for
    corr_min_scale: float = 0.60                  # lower bound of the scale factor
    critical_channels: tuple[str, ...] = DEFAULT_CRITICAL_CHANNELS
    critical_bad_fail_min: int = 2                # this many bad critical channels -> FAIL
    max_bad_channel_ratio: float = 0.10           # max bad channels = max(1, floor(ratio * n_eeg_channels))
    max_bad_channels: int | None = None           # exact count; overrides the ratio when set

    # ---- Layer 3: amplitude -------------------------------------------------------------------------
    typical_amp_min_uv: float = 1.0
    typical_amp_max_uv: float = 100.0
    high_amp_channel_fraction: float = 0.10
    dc_offset_info_uv: float = 100.0            # info only: |median| above this is listed as DC offset
    drift_info_uv: float = 50.0                  # info only: first-minute vs last-minute median shift

    # ---- Layer 4: spectral --------------------------------------------------------------------------
    psd_channels: tuple[str, ...] = ("O1", "O2", "Oz")       # posterior alpha ROI (absent channels are skipped)
    welch_window_s: float = 4.0
    welch_overlap: float = 0.5
    alpha_band_hz: tuple[float, float] = (8.0, 12.0)
    alpha_flank_hz: tuple[tuple[float, float], ...] = ((4.0, 7.0), (13.0, 16.0))
    alpha_prominence_min_db: float = 3.0
    alpha_ratio_pass: float = 1.5                 # Eyes Closed / Eyes Open alpha ratio below this -> FLAG
    alpha_no_blocking_ratio: float = 1.0          # no channel reaches this -> NO_ALPHA_BLOCKING
    dead_spectrum_band_hz: tuple[float, float] = (1.0, 45.0)
    dead_spectrum_power_uv2: float = 0.01
    line_freq: str = "auto"                       # "auto", "50" or "60" (explicit override of the sidecar)
    line_candidates_hz: tuple[float, ...] = (50.0, 60.0)
    line_detect_min_db: float = 6.0               # auto mode: a candidate must stand out by this much to be "detected"
    harmonic_orders: tuple[int, ...] = (2, 3)
    line_flank_offsets_hz: tuple[float, float] = (2.0, 5.0)
    line_peak_halfwidth_hz: float = 0.5
    line_flag_db: float = 20.0
    line_high_db: float = 30.0
    line_saturation_channel_fraction: float = 0.50
    notch_suggested_db: float = 10.0
    hf_cap_hz: float = 100.0                      # upper analysis bound at low sampling rates ...
    hf_nyquist_fraction: float = 0.90             # ... = min(hf_cap_hz, hf_nyquist_fraction * Nyquist)
    hf_rule_max_sfreq_hz: float = 250.0           # ... applied when sfreq <= this
    closed_tasks: tuple[str, ...] = ()            # explicit task/acq labels that mean Eyes Closed (override)
    open_tasks: tuple[str, ...] = ()              # explicit task/acq labels that mean Eyes Open (override)

    # ------------------------------------------------------------------------------------------ derived rules
    def max_bad_for(self, n_eeg: int) -> int:
        """Maximum number of bad (interpolatable) channels for a cap with ``n_eeg`` channels.

        ``max(1, floor(max_bad_channel_ratio * n_eeg))`` unless an exact count is configured. The product is
        rounded to 9 decimals before ``floor`` so that float noise (0.29 * 100 = 28.999999999999996) cannot
        lower the result.
        """
        if self.max_bad_channels is not None:
            return int(self.max_bad_channels)
        return max(1, math.floor(round(self.max_bad_channel_ratio * n_eeg, 9)))

    def corr_scale(self, n_eeg: int) -> float:
        """Factor applied to the spatial-correlation thresholds. 1.0 for caps of the reference density or more;
        ``sqrt(n / reference)`` (never below ``corr_min_scale``) for sparser caps, because the median of fewer,
        more widely spaced channels is a poorer reference for each channel."""
        if not self.corr_scale_by_channel_count:
            return 1.0
        return float(min(1.0, max(self.corr_min_scale, math.sqrt(n_eeg / self.corr_reference_n_channels))))

    def hf_limit(self, sfreq: float) -> float:
        """Upper edge (Hz) of the high-frequency analysis range. ``min(hf_cap_hz, 0.9 * Nyquist)`` when
        ``sfreq <= hf_rule_max_sfreq_hz``; otherwise unbounded (the configured band edges apply)."""
        if sfreq <= self.hf_rule_max_sfreq_hz:
            return float(min(self.hf_cap_hz, self.hf_nyquist_fraction * sfreq / 2.0))
        return float("inf")

    def band_hi(self, band_hi: float, sfreq: float) -> float:
        """A configured upper band edge, lowered where the sampling rate requires it."""
        return float(min(band_hi, self.hf_limit(sfreq)))

    # ------------------------------------------------------------------------------------------ (de)serialisation
    def validate(self) -> "QCConfig":
        if not 0 < self.max_bad_channel_ratio <= 1:
            raise ConfigError("max_bad_channel_ratio must be in (0, 1]")
        if self.max_bad_channels is not None and self.max_bad_channels < 0:
            raise ConfigError("max_bad_channels must be >= 0")
        if self.min_sfreq_hz <= 0:
            raise ConfigError("min_sfreq_hz must be positive")
        if self.max_sfreq_hz is not None and self.max_sfreq_hz < self.min_sfreq_hz:
            raise ConfigError("max_sfreq_hz must not be below min_sfreq_hz")
        if str(self.line_freq).lower() not in ("auto", "50", "60", "50.0", "60.0"):
            raise ConfigError("line_freq must be 'auto', '50' or '60'")
        if self.detrend_method not in ("highpass", "linear", "none"):
            raise ConfigError("detrend_method must be 'highpass', 'linear' or 'none'")
        if self.critical_bad_fail_min < 1:
            raise ConfigError("critical_bad_fail_min must be >= 1")
        if self.duration_ok_range_s[0] > self.duration_ok_range_s[1]:
            raise ConfigError("duration_ok_range_s must be (low, high)")
        return self

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=list))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QCConfig":
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ConfigError(f"unknown configuration key(s): {', '.join(unknown)}")
        return cls(**{k: _tuplify(v) for k, v in data.items()}).validate()

    def with_overrides(self, **overrides: Any) -> "QCConfig":
        merged = {**self.to_dict(), **{k: v for k, v in overrides.items() if v is not None}}
        if "line_freq" in merged:
            merged["line_freq"] = str(merged["line_freq"]).replace(".0", "")
        return QCConfig.from_dict(merged)

    def fingerprint(self) -> str:
        """Short hash of every threshold and the logic version. Results are reused only when it matches."""
        blob = json.dumps({"logic": QC_LOGIC_VERSION, "config": self.to_dict()}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def forced_line_hz(self) -> float | None:
        v = str(self.line_freq).lower()
        return None if v == "auto" else float(v)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``KEY=VALUE`` (VALUE is JSON, or a bare string) for ``--set``."""
    if "=" not in text:
        raise ConfigError(f"--set expects KEY=VALUE, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if key not in {f.name for f in fields(QCConfig)}:
        raise ConfigError(f"unknown configuration key: {key}")
    return key, value


def default_config() -> QCConfig:
    return QCConfig().validate()
