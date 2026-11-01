"""Regression against the validated prototype (``qc_eeg_subject.py``, QC version ``screening-2.2``).

The golden numbers below were produced by running the PROTOTYPE's functions and this package's engine on the same
synthetic 64-channel, 1 kHz recording (the prototype's native setting) and were verified to be identical (every column of
the channel table bit-identical). They guard the scientific logic against silent drift. They are valid for the default
``QCConfig``; at 64 channels and 1 kHz every adaptive rule of the package reduces to the prototype's constant.
"""
from __future__ import annotations

import pytest
from conftest import CAP_64, synth_uv

from openneuro_eeg_triage import engine as E
from openneuro_eeg_triage.config import QCConfig

SF = 1000.0


@pytest.fixture(scope="module")
def defective_64():
    cfg = QCConfig()
    data = synth_uv(CAP_64, sfreq=SF, duration=130, seed=3, flat=["F3"], noisy=["C3"], uncorrelated=["P3"], clip=["F4"],
                    line_hz=50, line_amp=6, alpha_amp=20)
    rec = E.Recording(list(CAP_64), list(CAP_64), SF, data, cfg=cfg)
    table = E.compute_channel_table(rec, cfg, E.spatial_correlation(rec, cfg))
    return cfg, rec, table


def test_adaptive_rules_reduce_to_prototype_constants_at_64_channels_1khz():
    c = QCConfig()
    assert c.max_bad_for(64) == 6 and c.corr_scale(64) == 1.0 and c.hf_limit(1000.0) == float("inf")
    assert (c.corr_threshold, c.corr_threshold_relaxed, c.critical_channels) == (0.20, 0.18, ("Cz", "Fz", "Pz", "O1", "O2"))
    assert round(c.frozen_min_s * 1000) == 100 and c.alpha_band_hz == (8.0, 12.0) and c.line_flag_db == 20 and c.line_high_db == 30


def test_layer2_matches_prototype(defective_64):
    cfg, rec, table = defective_64
    integ = E.integrity_check(rec, cfg, table)
    m = integ["metrics"]
    assert m["bad_channels"] == ["F3", "C3", "P3", "F4"]
    assert (m["n_flat"], m["n_clipped"], m["n_noisy"], m["n_uncorrelated"], m["n_critical_bad"]) == (1, 1, 1, 2, 0)
    assert m["median_spatial_corr"] == pytest.approx(0.9150065016364999, rel=1e-9)
    assert integ["verdict"] == "PASS" and {i["code"] for i in integ["issues"]} == {"USABLE_NEEDS_INTERPOLATION"}


def test_layer3_matches_prototype(defective_64):
    cfg, _, table = defective_64
    assert E.amplitude_check(cfg, table)["metrics"]["typical_amplitude_uv"] == pytest.approx(13.082240326829815, rel=1e-9)


def test_layer4_matches_prototype(defective_64):
    cfg, rec, _ = defective_64
    occ = E.occipital_metrics(E.compute_psd(rec, cfg), cfg, 50.0)
    assert occ["alpha_peak_hz"] == 10.0
    assert occ["alpha_prominence_db"] == pytest.approx(23.468974343312077, rel=1e-9)
    assert occ["alpha_power_uv2"] == pytest.approx(215.94190646708012, rel=1e-9)
    assert occ["line_db"] == pytest.approx(21.67116624734437, rel=1e-9)
    assert occ["broadband_power_uv2"] == pytest.approx(285.78538505360484, rel=1e-9)


@pytest.mark.parametrize("ec,eo,expected", [(dict(alpha_amp=30), dict(alpha_amp=3), 35.48905959895222),
                                            (dict(alpha_amp=0), dict(alpha_amp=0), 1.0886164364226412)])
def test_alpha_blocking_ratio_matches_prototype(ec, eo, expected):
    cfg, occs = QCConfig(), []
    for kw, seed in ((ec, 1), (eo, 2)):
        r = E.Recording(list(CAP_64), list(CAP_64), SF, synth_uv(CAP_64, sfreq=SF, duration=130, seed=seed, **kw), cfg=cfg)
        occs.append(E.occipital_metrics(E.compute_psd(r, cfg), cfg, 50.0))
    assert E.alpha_ratio_info(occs[0], occs[1])["median"] == pytest.approx(expected, rel=1e-9)
