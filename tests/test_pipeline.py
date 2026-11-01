"""Engine tests on synthetic recordings: the 5 layers, the standard-montage gatekeeper, adaptive thresholds."""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import CAP_19, CAP_19_LEGACY, CAP_32, CAP_64, CAPS, cap128, make_recording

from openneuro_eeg_triage.config import ConfigError, QCConfig, parse_override
from openneuro_eeg_triage.engine import (alpha_ratio_info, analyse_recording, check_montage, consistency_layer,
                                         corr_thresholds, finalize_records, infer_condition,
                                         normalize_channel_name, spectral_layer, subject_label)


# ------------------------------------------------------------------------------------------------ helpers
def run(names, cfg=None, sidecar=None, **kw):
    cfg = cfg or QCConfig()
    sf = kw.pop("sfreq", 128.0)
    return analyse_recording(make_recording(names, cfg, sfreq=sf, **kw), cfg, sidecar=sidecar)


def codes(layer_result):
    return {i["code"] for i in layer_result["issues"]}


def as_record(res, rec_id, task="rest", acq=None, run_=None, session="ses-1", cfg=None):
    """Wrap an analyse_recording result like the pipeline does, so finalize_records can be used."""
    cfg = cfg or QCConfig()
    cond, src = infer_condition(task, acq, cfg)
    return {"rec_id": rec_id, "subject": "sub-01", "session": session, "task": task, "acq": acq or "", "run": run_ or "",
            "format": "edf", "condition": cond, "cond_source": src, "error": None, "error_code": None,
            "pair_key": [session, "*" if src == "task" else task, "*" if src == "acq" else (acq or ""), run_ or ""],
            "status": res["status"], "layers": res["layers"], "occ": res["occ"], "line": res["line"],
            "clipped_frac": res["clipped_frac"], "summary": res["summary"], "n_eeg": res["n_eeg"], "info": res["info"]}


# ------------------------------------------------------------------------------------------------ config
def test_max_bad_channels_uses_floor_and_minimum_of_one():
    c = QCConfig()
    assert c.max_bad_for(64) == 6 and c.max_bad_for(65) == 6        # 6.5 -> 6 (floor, never round up)
    assert c.max_bad_for(32) == 3 and c.max_bad_for(128) == 12
    assert c.max_bad_for(19) == 1 and c.max_bad_for(5) == 1          # at least one
    assert QCConfig(max_bad_channel_ratio=0.29).max_bad_for(100) == 29    # float noise (28.999...) must not lower it
    assert QCConfig(max_bad_channels=5).max_bad_for(64) == 5          # exact count overrides the ratio


def test_config_validation_overrides_and_fingerprint():
    base = QCConfig()
    assert base.fingerprint() == QCConfig().fingerprint()
    assert base.with_overrides(noisy_ratio=5.0).fingerprint() != base.fingerprint()
    with pytest.raises(ConfigError):
        QCConfig(max_bad_channel_ratio=0.0).validate()
    with pytest.raises(ConfigError):
        QCConfig.from_dict({"not_a_key": 1})
    assert parse_override("noisy_ratio=5") == ("noisy_ratio", 5)
    assert parse_override("critical_channels=[\"Cz\"]")[1] == ["Cz"]
    assert QCConfig.from_dict({"critical_channels": ["Cz", "Pz"]}).critical_channels == ("Cz", "Pz")


def test_adaptive_bounds():
    c = QCConfig()
    assert c.corr_scale(64) == 1.0 and c.corr_scale(128) == 1.0
    assert c.corr_scale(32) == pytest.approx(math.sqrt(0.5)) and c.corr_scale(19) == c.corr_min_scale
    assert not QCConfig(corr_scale_by_channel_count=False).corr_scale(19) < 1
    assert c.hf_limit(1000) == math.inf and c.band_hi(45, 1000) == 45
    assert c.hf_limit(250) == 100.0 and c.hf_limit(128) == pytest.approx(57.6)
    assert c.band_hi(45, 64) == pytest.approx(28.8)                   # 0.9 * Nyquist when sfreq is very low
    thr19, thr64 = corr_thresholds(CAP_19, c), corr_thresholds(CAP_64, c)
    assert thr19[CAP_19.index("Cz")] == pytest.approx(0.20 * 0.6) and thr64[CAP_64.index("Cz")] == pytest.approx(0.20)
    assert thr64[CAP_64.index("O1")] == pytest.approx(0.18)           # relaxed occipital threshold preserved


# ------------------------------------------------------------------------------------------------ names & montage
@pytest.mark.parametrize("raw,expected", [
    ("EEG Fz-REF", "Fz"), ("EEG O1-LE", "O1"), ("fp1", "Fp1"), ("oz", "Oz"), ("fz", "Fz"), ("CZ", "Cz"), ("pz", "Pz"),
    ("T3", "T7"), ("T4", "T8"), ("T5", "P7"), ("T6", "P8"), ("t3", "T7"), ("AFZ", "AFz"), ("Cz.", "Cz"), ("Pz-A1", "Pz")])
def test_channel_name_normalisation(raw, expected):
    assert normalize_channel_name(raw) == (expected, True)


@pytest.mark.parametrize("raw", ["E12", "A5", "B17", "STATUS", "Ch1", "EEG"])
def test_unrecognised_names_are_never_guessed(raw):
    name, ok = normalize_channel_name(raw)
    assert not ok and name == raw


@pytest.mark.parametrize("names", [CAP_19, CAP_19_LEGACY, CAP_32, CAP_64, cap128(), [f"EEG {n}-REF" for n in CAP_32],
                                   [n.lower() for n in CAP_32]])
def test_standard_montages_are_supported(names):
    assert check_montage(names, QCConfig()).supported


@pytest.mark.parametrize("names", [[f"E{i}" for i in range(1, 129)], [f"A{i}" for i in range(1, 33)],
                                   [f"{c}{i}" for c in "AB" for i in range(1, 33)], [f"Ch{i}" for i in range(1, 20)]])
def test_non_standard_montages_are_rejected(names):
    chk = check_montage(names, QCConfig())
    assert not chk.supported and chk.reason


def test_real_dortmund_channels_tsv_is_accepted():
    """The real channels.tsv of OpenNeuro ds005385 (CC0 metadata, 64 labels) passes the standard-montage gatekeeper."""
    from pathlib import Path

    from openneuro_eeg_triage.engine import eeg_names_from_channels_tsv
    names, types = eeg_names_from_channels_tsv(Path(__file__).parent / "data" / "ds005385_channels.tsv")
    chk = check_montage(names, QCConfig())
    assert len(names) == 64 and chk.supported and chk.n_recognized == 64 and len(chk.anchors_found) == 10


def test_legacy_names_are_mapped_and_reported():
    chk = check_montage(CAP_19_LEGACY, QCConfig())
    assert set(chk.legacy_mapped) == {"T3->T7", "T4->T8", "T5->P7", "T6->P8"}


# ------------------------------------------------------------------------------------------------ clean recordings, any cap
@pytest.mark.parametrize("n", [19, 32, 64])
def test_clean_recording_passes_layers_1_to_3(n):
    res = run(CAPS[n])
    assert res["status"] == "OK"
    for layer in ("header", "integrity", "amplitude"):
        assert res["layers"][layer]["verdict"] == "PASS", (layer, res["layers"][layer]["issues"])
    m = res["layers"]["integrity"]["metrics"]
    assert m["n_bad_channels"] == 0
    assert m["max_bad_allowed"] == max(1, math.floor(0.10 * n))      # adapts to the cap, no n == 64 lock
    assert res["layers"]["header"]["metrics"]["n_eeg_channels"] == n


def test_128_channel_cap_runs():
    res = run(cap128(), duration=130)
    assert res["status"] == "OK" and res["layers"]["integrity"]["metrics"]["max_bad_allowed"] == 12


def test_cap_without_oz_falls_back_to_o1_o2():
    res = run(CAP_19)
    assert "Oz" not in CAP_19 and res["layers"]["header"]["verdict"] == "PASS"       # Oz is not required
    assert res["layers"]["header"]["metrics"]["posterior_roi"] == ["O1", "O2"]
    assert res["occ"]["channels"] == ["O1", "O2"]
    res32 = run(CAP_32)
    assert res32["occ"]["channels"] == ["O1", "O2", "Oz"]


def test_legacy_19_channel_names_work_end_to_end():
    res = run(CAP_19_LEGACY)
    assert res["status"] == "OK"
    assert "T7" in res["summary"]["ch_names"] and "T3" not in res["summary"]["ch_names"]
    assert res["layers"]["header"]["verdict"] == "PASS"


def test_required_channel_missing_fails():
    names = [c for c in CAP_32 if c != "O2"]
    res = run(names)
    assert "REQUIRED_CHANNELS_MISSING" in codes(res["layers"]["header"]) and res["layers"]["header"]["verdict"] == "FAIL"


# ------------------------------------------------------------------------------------------------ layer 1: header
def test_header_limits():
    assert "DURATION_TOO_SHORT" in codes(run(CAP_32, duration=50)["layers"]["header"])
    short = run(CAP_32, duration=90)["layers"]["header"]
    assert codes(short) == {"DURATION_OUTSIDE_RANGE"} and short["verdict"] == "FLAG"
    assert "SFREQ_TOO_LOW" in codes(run(CAP_32, sfreq=100.0)["layers"]["header"])
    assert "SFREQ_TOO_LOW" not in codes(run(CAP_32, cfg=QCConfig(min_sfreq_hz=100.0), sfreq=100.0)["layers"]["header"])
    assert "SFREQ_ABOVE_MAX" in codes(run(CAP_32, cfg=QCConfig(max_sfreq_hz=100.0))["layers"]["header"])
    twelve = ["Fz", "Cz", "Pz", "Oz", "O1", "O2", "F3", "F4", "C3", "C4", "Fp1", "Fp2"]
    assert "TOO_FEW_EEG_CHANNELS" in codes(run(twelve)["layers"]["header"])


def test_sidecar_checks():
    ok = run(CAP_32, sidecar={"SamplingFrequency": 128, "EEGChannelCount": 32})["layers"]["header"]
    assert ok["verdict"] == "PASS"
    bad = run(CAP_32, sidecar={"SamplingFrequency": 256, "EEGChannelCount": 31})["layers"]["header"]
    assert codes(bad) == {"SIDECAR_SAMPLING_RATE_MISMATCH", "SIDECAR_CHANNEL_COUNT_MISMATCH"} and bad["verdict"] == "FLAG"


@pytest.mark.parametrize("names", [[f"E{i}" for i in range(1, 33)], [f"A{i}" for i in range(1, 33)]])
def test_non_standard_montage_recording_is_rejected_cleanly(names):
    res = run(names)                                     # must not raise
    assert res["status"] == "UNSUPPORTED" and res["ch_table"] is None
    header = res["layers"]["header"]
    assert header["verdict"] == "FAIL" and "UNSUPPORTED_MONTAGE" in codes(header)
    rec = finalize_records([as_record(res, "r1")], QCConfig())[0]
    assert rec["overall"] == "FAIL" and rec["label"] == "REJECT" and rec["codes"] == ["UNSUPPORTED_MONTAGE"]


# ------------------------------------------------------------------------------------------------ layer 2: integrity
def test_few_bad_channels_are_repairable_not_rejected():
    res = run(CAP_64, flat=["F3", "P4"])
    integ = res["layers"]["integrity"]
    assert integ["verdict"] == "PASS" and "USABLE_NEEDS_INTERPOLATION" in codes(integ)
    assert integ["metrics"]["bad_channels"] == ["F3", "P4"] and integ["metrics"]["n_flat"] == 2
    assert finalize_records([as_record(res, "r")], QCConfig())[0]["label"] == "USABLE_NEEDS_INTERPOLATION"


def test_one_critical_channel_is_a_flag_two_are_a_fail():
    one = run(CAP_64, flat=["Cz"])["layers"]["integrity"]
    assert "CRITICAL_CHANNEL_BAD" in codes(one) and one["verdict"] == "FLAG"
    two = run(CAP_64, flat=["O1", "O2"])["layers"]["integrity"]
    assert "CRITICAL_CHANNELS_BAD" in codes(two) and two["verdict"] == "FAIL"


def test_too_many_bad_channels_scales_with_the_cap():
    four_flat = ["F3", "F4", "C3", "C4"]
    assert "TOO_MANY_BAD_CHANNELS" in codes(run(CAP_32, flat=four_flat)["layers"]["integrity"])      # max 3 of 32
    assert run(CAP_64, flat=four_flat)["layers"]["integrity"]["verdict"] == "PASS"                    # max 6 of 64
    assert "TOO_MANY_BAD_CHANNELS" in codes(run(CAP_19, flat=["F3", "F4"])["layers"]["integrity"])     # max 1 of 19


def test_cli_style_overrides_of_the_bad_channel_limit():
    four_flat = ["F3", "F4", "C3", "C4"]
    assert run(CAP_32, cfg=QCConfig(max_bad_channels=5), flat=four_flat)["layers"]["integrity"]["verdict"] == "PASS"
    assert run(CAP_32, cfg=QCConfig(max_bad_channel_ratio=0.25), flat=four_flat)["layers"]["integrity"]["metrics"]["max_bad_allowed"] == 8
    assert "TOO_MANY_BAD_CHANNELS" in codes(run(CAP_64, cfg=QCConfig(max_bad_channels=1), flat=["F3", "F4"])["layers"]["integrity"])


def test_noisy_and_uncorrelated_channels_are_found():
    res = run(CAP_32, noisy=["C3"], uncorrelated=["P3"])
    m = res["layers"]["integrity"]["metrics"]
    assert "C3" in m["noisy_channels"] and "P3" in m["uncorrelated_channels"]
    assert m["n_bad_channels"] == 2


def test_clipped_channel_is_found():
    assert "F3" in run(CAP_32, clip=["F3"])["layers"]["integrity"]["metrics"]["clipped_channels"]


def test_unreliable_correlation_reference_flags_instead_of_marking_everyone_bad():
    res = run(CAP_32, uncorrelated=CAP_32[:26])       # most channels unrelated to each other: the median is meaningless
    integ = res["layers"]["integrity"]
    assert "LOW_GLOBAL_SPATIAL_CORRELATION" in codes(integ) and integ["metrics"]["n_bad_channels"] == 0
    assert integ["verdict"] == "FLAG"


def test_frozen_segments():
    assert codes(run(CAP_32, frozen=(30, 8))["layers"]["integrity"]) >= {"FROZEN_SEGMENT"}
    long_freeze = run(CAP_32, frozen=(20, 50))["layers"]["integrity"]
    assert "RECORDING_FROZEN" in codes(long_freeze) and long_freeze["verdict"] == "FAIL"


def test_baseline_drift_does_not_create_bad_channels():
    names = CAP_32
    rec = make_recording(names)
    t = np.arange(rec.n_times) / rec.sfreq
    rec.data_uv = rec.data_uv + 3000 * np.sin(2 * np.pi * 0.02 * t)[None, :] + np.linspace(0, 400, len(names))[:, None]
    res = analyse_recording(rec, QCConfig())
    assert res["layers"]["integrity"]["metrics"]["n_bad_channels"] == 0
    assert res["layers"]["amplitude"]["verdict"] == "PASS"
    assert res["layers"]["amplitude"]["metrics"]["n_drift_info"] >= 0 and res["layers"]["amplitude"]["metrics"]["n_dc_offset_info"] > 0


def test_raw_data_are_never_modified():
    rec = make_recording(CAP_32)
    before = rec.data_uv.copy()
    analyse_recording(rec, QCConfig())
    assert np.array_equal(before, rec.data_uv)


# ------------------------------------------------------------------------------------------------ layer 3: amplitude
def test_amplitude_out_of_range_is_a_fail():
    assert "AMPLITUDE_OUT_OF_RANGE" in codes(run(CAP_32, scale=1e-3)["layers"]["amplitude"])       # volts mislabeled as uV
    big = run(CAP_32, scale=30)["layers"]["amplitude"]
    assert "AMPLITUDE_OUT_OF_RANGE" in codes(big) and big["verdict"] == "FAIL"


# ------------------------------------------------------------------------------------------------ layer 4: spectral
def test_line_frequency_auto_detection_50_and_60():
    r50 = run(CAP_32, line_hz=50, line_amp=8)
    r60 = run(CAP_32, line_hz=60, line_amp=8)
    assert (r50["line"]["hz"], r50["line"]["source"]) == (50.0, "auto")
    assert (r60["line"]["hz"], r60["line"]["source"]) == (60.0, "auto")
    for r, hz in ((r50, 50), (r60, 60)):
        sp = finalize_records([as_record(r, "x")], QCConfig())[0]["layers"]["spectral"]
        assert "LINE_NOISE" in codes(sp) or "HIGH_LINE_NOISE" in codes(sp)
        assert sp["metrics"]["line_hz"] == hz and sp["metrics"]["notch_suggested"] is True


def test_line_frequency_priority_cli_over_sidecar_over_auto():
    r = run(CAP_32, line_hz=60, line_amp=8, sidecar={"PowerLineFrequency": 60})
    assert (r["line"]["hz"], r["line"]["source"]) == (60.0, "sidecar")
    r = run(CAP_32, line_hz=60, line_amp=8, sidecar={"PowerLineFrequency": "n/a"})
    assert r["line"]["source"] == "auto"                              # unusable sidecar value -> spectrum
    forced = run(CAP_32, cfg=QCConfig(line_freq="50"), line_hz=60, line_amp=8, sidecar={"PowerLineFrequency": 60})
    assert (forced["line"]["hz"], forced["line"]["source"]) == (50.0, "config")


def test_clean_recording_has_no_line_flag_and_reports_default():
    r = run(CAP_32)
    assert r["line"]["source"] == "default_no_line_detected"
    sp = finalize_records([as_record(r, "x")], QCConfig())[0]["layers"]["spectral"]
    assert not codes(sp) & {"LINE_NOISE", "HIGH_LINE_NOISE", "LINE_NOISE_SATURATION"}


def test_line_noise_is_a_fail_only_together_with_saturation():
    cfg = QCConfig()
    occ = {"channels": ["O1", "O2"], "alpha_prominence_db": 6.0, "alpha_peak_hz": 10.0, "alpha_power_uv2": 10.0,
           "line_hz": 50.0, "line_db": 35.0, "broadband_power_uv2": 50.0, "dead_spectrum": False,
           "line_h2_db": 5.0, "line_h3_db": 4.0, "alpha_power_uv2_O1": 10.0, "alpha_power_uv2_O2": 10.0}
    flag = spectral_layer(occ, "unknown", {"applicable": False}, 0.1, cfg)
    assert flag["verdict"] == "FLAG" and codes(flag) == {"HIGH_LINE_NOISE"}
    fail = spectral_layer(occ, "unknown", {"applicable": False}, 0.6, cfg)
    assert fail["verdict"] == "FAIL" and codes(fail) == {"LINE_NOISE_SATURATION"}
    occ["line_db"] = 25.0
    assert codes(spectral_layer(occ, "unknown", {"applicable": False}, 0.6, cfg)) == {"LINE_NOISE"}


def test_line_detection_on_a_very_low_sampling_rate_does_not_crash():
    cfg = QCConfig(min_sfreq_hz=64.0)
    r = run(CAP_32, cfg=cfg, sfreq=100.0, line_hz=50, line_amp=8)     # 50 Hz is at Nyquist: not assessable
    assert r["status"] == "OK" and r["line"]["source"] == "not_assessable" and r["line"]["hz"] is None
    sp = finalize_records([as_record(r, "x", cfg=cfg)], cfg)[0]["layers"]["spectral"]
    assert math.isnan(sp["metrics"]["line_db"]) and not codes(sp) & {"LINE_NOISE", "HIGH_LINE_NOISE"}


def test_dead_posterior_spectrum_fails():
    res = run(CAP_32, flat=["O1", "O2", "Oz"])
    assert res["occ"]["dead_spectrum"] is True
    rec = finalize_records([as_record(res, "x")], QCConfig())[0]
    assert "DEAD_SPECTRUM" in codes(rec["layers"]["spectral"]) and rec["overall"] == "FAIL" and rec["label"] == "REJECT"


def test_alpha_blocking_is_measured_from_closed_open_pairs():
    ec = as_record(run(CAP_32, alpha_amp=30), "ec", task="EyesClosed")
    eo = as_record(run(CAP_32, alpha_amp=3, seed=1), "eo", task="EyesOpen")
    out = {r["rec_id"]: r for r in finalize_records([ec, eo], QCConfig())}
    for r in out.values():
        assert r["layers"]["spectral"]["metrics"]["alpha_ratio_EC_EO"] > 1.5
        assert not codes(r["layers"]["spectral"]) & {"WEAK_ALPHA_BLOCKING", "NO_ALPHA_BLOCKING", "LOW_ALPHA_PROMINENCE"}
        assert r["overall"] == "PASS" and r["label"] == "USABLE"


def test_alpha_non_responder_is_flag_never_fail():
    ec = as_record(run(CAP_32, alpha_amp=0), "ec", task="EyesClosed")
    eo = as_record(run(CAP_32, alpha_amp=0, seed=1), "eo", task="EyesOpen")
    out = {r["rec_id"]: r for r in finalize_records([ec, eo], QCConfig())}
    sp = out["ec"]["layers"]["spectral"]
    assert "LOW_ALPHA_PROMINENCE" in codes(sp) and codes(sp) & {"WEAK_ALPHA_BLOCKING", "NO_ALPHA_BLOCKING"}
    assert sp["metrics"]["alpha_non_responder_possible"] is True
    assert out["ec"]["overall"] == "FLAG" and out["ec"]["label"] == "USABLE_WITH_FLAGS"
    assert "LOW_ALPHA_PROMINENCE" not in codes(out["eo"]["layers"]["spectral"])      # prominence is judged on Eyes Closed only


def test_unknown_eye_condition_gets_no_alpha_flags():
    res = as_record(run(CAP_32, alpha_amp=0), "rest1", task="rest")
    r = finalize_records([res], QCConfig())[0]
    assert r["condition"] == "unknown" and r["overall"] == "PASS"
    assert math.isnan(r["layers"]["spectral"]["metrics"]["alpha_ratio_EC_EO"])


def test_closed_only_design_is_not_flagged_for_a_missing_partner():
    r = finalize_records([as_record(run(CAP_32), "ec", task="EyesClosed")], QCConfig())[0]
    assert "ALPHA_RATIO_NOT_COMPUTED" not in r["codes"]


def test_unpaired_closed_and_open_recordings_are_flagged():
    ec = as_record(run(CAP_32), "ec", task="EyesClosed", run_="1")
    eo = as_record(run(CAP_32, seed=1), "eo", task="EyesOpen", run_="2")             # different run: no partner
    out = finalize_records([ec, eo], QCConfig())
    assert all("ALPHA_RATIO_NOT_COMPUTED" in r["codes"] for r in out)


def test_pairing_on_acq_entity_and_session():
    recs = [as_record(run(CAP_32, alpha_amp=30 if c == "closed" else 3, seed=i), f"{ses}_{c}", task="rest",
                      acq="EC" if c == "closed" else "EO", session=ses)
            for i, (ses, c) in enumerate([("ses-1", "closed"), ("ses-1", "open"), ("ses-2", "closed"), ("ses-2", "open")])]
    out = finalize_records(recs, QCConfig())
    assert all(r["condition"] in ("closed", "open") for r in out)
    assert all(r["layers"]["spectral"]["metrics"]["alpha_ratio_EC_EO"] > 1.5 for r in out)


def test_explicit_task_overrides_for_eye_condition():
    cfg = QCConfig(closed_tasks=("rest1",), open_tasks=("rest2",))
    assert infer_condition("rest1", None, cfg)[0] == "closed" and infer_condition("rest2", None, cfg)[0] == "open"
    assert infer_condition("rest", None, QCConfig())[0] == "unknown"


@pytest.mark.parametrize("task,acq,expected", [("EyesClosed", "pre", "closed"), ("EyesOpen", None, "open"), ("restEC", None, "closed"),
                                               ("eo", None, "open"), ("rest", "EC", "closed"), ("rest", "open", "open"),
                                               ("rest", None, "unknown"), ("Resting", None, "unknown"), ("oddball", None, "unknown")])
def test_eye_condition_inference(task, acq, expected):
    assert infer_condition(task, acq, QCConfig())[0] == expected


def test_alpha_ratio_info_handles_missing_partner():
    info = alpha_ratio_info(None, None)
    assert info["applicable"] and info["valid"] == [] and math.isnan(info["median"])


# ------------------------------------------------------------------------------------------------ layer 5: consistency
def test_consistency_flags_the_deviating_recording():
    same = {"ch_names": CAP_32, "sfreq": 128.0}
    shuffled = {"ch_names": CAP_32[::-1], "sfreq": 128.0}
    other_sf = {"ch_names": CAP_32, "sfreq": 256.0}
    out = consistency_layer({"a": same, "b": same, "c": shuffled, "d": other_sf})
    assert out["a"]["verdict"] == "PASS" and out["b"]["verdict"] == "PASS"
    assert codes(out["c"]) == {"CHANNEL_LAYOUT_DIFFERS"} and codes(out["d"]) == {"SFREQ_DIFFERS"}
    assert consistency_layer({}) == {}


def test_consistency_in_finalize_records():
    recs = [as_record(run(CAP_32, seed=i), f"r{i}", task="rest", run_=str(i)) for i in range(2)]
    recs.append(as_record(run(CAP_32[::-1], seed=9), "r9", task="rest", run_="9"))
    out = {r["rec_id"]: r for r in finalize_records(recs, QCConfig())}
    assert out["r9"]["overall"] == "FAIL" and "CHANNEL_LAYOUT_DIFFERS" in out["r9"]["codes"]
    assert out["r0"]["layers"]["consistency"]["verdict"] == "PASS"


# ------------------------------------------------------------------------------------------------ labels
def test_missing_and_error_records():
    rec_missing = {"rec_id": "m", "status": "MISSING", "layers": {}, "condition": "unknown", "pair_key": [], "error": "x",
                   "error_code": "DOWNLOAD_FAILED", "summary": {}}
    rec_err = {"rec_id": "e", "status": "ERROR", "layers": {}, "condition": "unknown", "pair_key": [], "error": "bad file",
               "error_code": "FILE_NOT_READABLE", "summary": {}}
    m, e = finalize_records([rec_missing, rec_err], QCConfig())
    assert (m["overall"], m["label"], m["codes"]) == ("MISSING", "MISSING", ["DOWNLOAD_FAILED"])
    assert (e["overall"], e["label"], e["codes"]) == ("FAIL", "REJECT", ["FILE_NOT_READABLE"])


def test_subject_labels():
    assert subject_label(["USABLE", "USABLE"]) == "USABLE"
    assert subject_label(["USABLE", "USABLE_NEEDS_INTERPOLATION"]) == "USABLE_NEEDS_INTERPOLATION"
    assert subject_label(["USABLE", "REJECT"]) == "PARTIALLY_USABLE"
    assert subject_label(["REJECT", "REJECT"]) == "REJECT"
    assert subject_label(["MISSING"]) == "NO DATA" and subject_label([]) == "NO DATA"
