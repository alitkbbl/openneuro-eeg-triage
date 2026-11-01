"""End-to-end tests of the pure-CLI workflow on synthetic datasets (local BIDS and a fake OpenNeuro server)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from conftest import build_bids, ec_eo, have_soffice

from openneuro_eeg_triage import pipeline
from openneuro_eeg_triage.cli import build_parser, main
from openneuro_eeg_triage.reporter import SHEETS

SRC = Path(__file__).resolve().parents[1] / "src" / "openneuro_eeg_triage"


def cli(argv, capsys):
    code = main([str(a) for a in argv])
    cap = capsys.readouterr()
    return code, cap.out + cap.err


def sheet(out: Path, name: str) -> pd.DataFrame:
    return pd.read_excel(out / "qc_summary.xlsx", sheet_name=name)


def all_files(root: Path):
    return [p for p in Path(root).rglob("*") if p.is_file()] if Path(root).exists() else []


# ------------------------------------------------------------------------------------------------ a 30-subject cohort
def cohort(n=30):
    subs = []
    for i in range(1, n + 1):
        cap = 64 if i % 5 == 0 else (19 if i % 2 else 32)         # 19, 32 and 64 channel caps in ONE dataset
        sig: dict = {}
        if i % 7 == 0:
            sig["flat"] = ["F3", "P4"]                            # repairable on 32/64 ch, too many on a 19-ch cap (max 1)
        if i % 11 == 0:
            sig["flat"] = ["O1", "O2"]                            # critical channels: rejected
        if i % 3 == 0:
            sig.update(line_hz=50, line_amp=8)
        if i % 4 == 0:
            sig.update(line_hz=60, line_amp=8)
        subs.append({"id": f"{i:03d}", "cap": cap, "recordings": ec_eo(**sig), "age": 20 + 2 * i,
                     "sidecar_line": 60 if i % 4 == 0 else (50 if i % 3 == 0 else None)})
    return subs


@pytest.fixture(scope="module")
def cohort_run(tmp_path_factory):
    """Run the CLI twice on one output directory: subjects 1-20, then 21-30. Snapshots after each run."""
    base = tmp_path_factory.mktemp("cohort")
    bids = build_bids(base / "bids", cohort(), "Synthetic cohort")
    out = base / "out"
    base_args = ["--bids-dir", str(bids), "-o", str(out), "--no-plots", "-q"]
    code1 = main([*base_args, "--subjects", "1", "to", "20"])
    snap1 = {"subjects": sheet(out, "Subjects"), "rank": sheet(out, "Recording_Ranking"), "code": code1}
    shutil.copytree(out, base / "out_after_run1")
    code2 = main([*base_args, "--subjects", "21", "to", "30"])
    snap2 = {"subjects": sheet(out, "Subjects"), "rank": sheet(out, "Recording_Ranking"), "code": code2}
    return {"base": base, "bids": bids, "out": out, "args": base_args, "snap1": snap1, "snap2": snap2}


def test_first_batch_is_screened(cohort_run):
    s1 = cohort_run["snap1"]
    assert s1["code"] == 0 and len(s1["subjects"]) == 20 and len(s1["rank"]) == 40


def test_second_batch_is_merged_into_one_complete_workbook(cohort_run):
    s2, out = cohort_run["snap2"], cohort_run["out"]
    assert s2["code"] == 0
    subj, rank = s2["subjects"], s2["rank"]
    assert len(subj) == 30 and len(rank) == 60                                           # 20 + 10 subjects, 40 + 20 recordings
    assert sorted(subj["subject"]) == sorted(f"sub-{i:03d}" for i in range(1, 31))
    wb = openpyxl.load_workbook(out / "qc_summary.xlsx")
    assert wb.sheetnames == SHEETS
    assert len(pd.read_csv(out / "qc_results.csv")) == 60 and len(pd.read_csv(out / "qc_subjects.csv")) == 30
    summary = json.loads((out / "qc_summary.json").read_text())
    assert summary["n_subjects"] == 30 and summary["n_recordings"] == 60
    events = [json.loads(line)["event"] for line in (out / "audit_trail.jsonl").read_text().splitlines()]
    assert events.count("run_start") == 2 and events.count("screened") == 60


def test_subjects_and_recordings_are_globally_reranked(cohort_run):
    s1, s2 = cohort_run["snap1"], cohort_run["snap2"]
    subj, rank = s2["subjects"], s2["rank"]
    assert sorted(subj["rank"]) == list(range(1, 31)) and rank["rec_rank"].tolist() == list(range(1, 61))
    # usable recordings first, ordered by score; rejected ones last
    usable = rank["label"].isin(["USABLE", "USABLE_WITH_FLAGS", "USABLE_NEEDS_INTERPOLATION"])
    assert usable.tolist() == sorted(usable.tolist(), reverse=True)
    scores = rank.loc[usable, "quality_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    # the best recordings of the 30 subjects are not simply "batch 1 first": batch-2 recordings interleave
    num = rank["subject"].str[4:].astype(int)
    usable_batch1_worst = rank.loc[usable & (num <= 20), "rec_rank"].max()
    usable_batch2_best = rank.loc[usable & (num > 20), "rec_rank"].min()
    assert usable_batch2_best < usable_batch1_worst
    # ranks of the earlier subjects were recomputed against the new ones
    old = s1["rank"].set_index(["subject", "task"])["rec_rank"]
    new = rank[rank["subject"].str[4:].astype(int) <= 20].set_index(["subject", "task"])["rec_rank"]
    assert (old != new.reindex(old.index)).any()
    # subject ranking uses the same global scale
    best = subj.sort_values("rank").iloc[0]
    assert best["quality_score"] == subj["quality_score"].max()


def test_synthetic_defects_end_up_in_the_right_labels(cohort_run):
    subj = cohort_run["snap2"]["subjects"].set_index("subject")
    assert subj.loc["sub-011", "label"] == "REJECT"                       # O1+O2 flat (critical)
    assert subj.loc["sub-007", "label"] == "REJECT"                       # 19-ch cap: 2 flat > max 1 bad channel
    assert subj.loc["sub-014", "label"] == "USABLE_NEEDS_INTERPOLATION"  # 32-ch cap: 2 flat <= max 3
    assert subj.loc["sub-001", "label"] in ("USABLE", "USABLE_WITH_FLAGS")
    assert subj.loc["sub-001", "n_eeg_channels"] == 19 and subj.loc["sub-005", "n_eeg_channels"] == 64
    rec = cohort_run["snap2"]["rank"]
    lines = rec.set_index("subject")
    assert (lines.loc["sub-012", "codes"] if isinstance(lines.loc["sub-012", "codes"], str) else "; ".join(lines.loc["sub-012", "codes"])).find("LINE_NOISE") >= 0


def test_line_frequency_50_and_60_are_detected_per_recording(cohort_run):
    full = pd.read_csv(cohort_run["out"] / "qc_results.csv")
    s4, s3 = full[full["subject"] == "sub-004"], full[full["subject"] == "sub-003"]
    assert set(s4["line_hz"]) == {60.0} and set(s4["line_freq_source"]) == {"sidecar"}
    assert set(s3["line_hz"]) == {50.0}


@pytest.mark.skipif(not have_soffice(), reason="LibreOffice not available")
def test_overview_formulas_recalculate_to_the_global_numbers(cohort_run, tmp_path):
    out = cohort_run["out"]
    conv = tmp_path / "conv"
    subprocess.run(["soffice", "--headless", "--convert-to", "xlsx", "--outdir", str(conv), str(out / "qc_summary.xlsx")],
                   check=True, capture_output=True, timeout=300)
    ws = openpyxl.load_workbook(conv / "qc_summary.xlsx", data_only=True)["Overview"]
    values = {ws.cell(r, 1).value: ws.cell(r, 2).value for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value}
    assert values["Subjects in the workbook"] == 30 and values["Recordings screened"] == 60
    assert values["  status COMPLETED"] == 30 and values["PASS"] + values["FLAG"] + values["FAIL"] == 60
    errors = [c.value for row in ws.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("#")]
    assert errors == []


def test_overview_has_formulas_and_charts_and_legend_names_the_mode(cohort_run):
    wb = openpyxl.load_workbook(cohort_run["out"] / "qc_summary.xlsx")
    ov = wb["Overview"]
    formulas = [c.value for row in ov.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("=")]
    assert len(formulas) > 40 and any("COUNTIF(Subjects!" in f for f in formulas)
    assert len(ov._charts) >= 5
    legend = " ".join(str(c.value) for row in wb["Legend"].iter_rows() for c in row if c.value)
    assert "Download–Screen–Delete (single-recording peak disk footprint)" in legend and "UNSUPPORTED_MONTAGE" in legend
    assert wb["Subjects"].freeze_panes and wb["Issues"].max_row > 1
    assert wb["Subjects"]["A1"].font.name == "Arial"


def test_resume_skips_everything_already_done(cohort_run, tmp_path, capsys):
    out = tmp_path / "out"
    shutil.copytree(cohort_run["out"], out)
    code, text = cli(["--bids-dir", cohort_run["bids"], "-o", out, "--no-plots"], capsys)
    assert code == 0 and text.count("already analysed, skipped") == 30 and "30 skipped" in text
    assert len(sheet(out, "Subjects")) == 30


def test_force_reanalyses_only_the_requested_subjects_and_keeps_the_rest(cohort_run, tmp_path, capsys):
    out = tmp_path / "out"
    shutil.copytree(cohort_run["out"], out)
    code, text = cli(["--bids-dir", cohort_run["bids"], "-o", out, "--no-plots", "--subjects", "3", "--force"], capsys)
    assert code == 0 and "1 subject(s) screened, 0 skipped" in text
    assert len(sheet(out, "Subjects")) == 30                              # the other 29 are still in the workbook


def test_overwrite_all_starts_from_scratch(cohort_run, tmp_path, capsys):
    out = tmp_path / "out"
    shutil.copytree(cohort_run["out"], out)
    code, text = cli(["--bids-dir", cohort_run["bids"], "-o", out, "--no-plots", "--subjects", "1", "to", "5", "--overwrite-all"], capsys)
    assert code == 0 and "removed the previous results of 30" in text
    assert len(sheet(out, "Subjects")) == 5 and not (out / "sub-030").exists()


def test_changed_thresholds_are_reanalysed_and_older_results_marked_stale(cohort_run, tmp_path, capsys):
    out = tmp_path / "out"
    shutil.copytree(cohort_run["out"], out)
    code, text = cli(["--bids-dir", cohort_run["bids"], "-o", out, "--no-plots", "--subjects", "7", "--max-bad-channels", "2"], capsys)
    assert code == 0 and "different thresholds" in text
    subj = sheet(out, "Subjects").set_index("subject")
    assert subj.loc["sub-007", "label"] != "REJECT"                       # 2 bad channels are now allowed on the 19-ch cap
    assert subj["config_stale"].sum() == 29 and not subj.loc["sub-007", "config_stale"]


def test_new_tasks_for_a_known_subject_extend_the_subject(cohort_run, tmp_path, capsys):
    out = tmp_path / "out"
    code, _ = cli(["--bids-dir", cohort_run["bids"], "-o", out, "--no-plots", "--subjects", "2", "--tasks", "EyesClosed"], capsys)
    assert code == 0 and sheet(out, "Subjects").loc[0, "n_recordings"] == 1
    code, _ = cli(["--bids-dir", cohort_run["bids"], "-o", out, "--no-plots", "--subjects", "2", "--tasks", "EyesOpen"], capsys)
    row = sheet(out, "Subjects").loc[0]
    assert code == 0 and row["n_recordings"] == 2 and len(sheet(out, "Recordings")) == 2


def test_output_directory_cannot_silently_mix_datasets(cohort_run, tmp_path, capsys):
    other = build_bids(tmp_path / "other", [{"id": "001", "cap": 19, "recordings": [{"task": "rest"}]}])
    code, text = cli(["--bids-dir", other, "-o", cohort_run["out"], "--no-plots"], capsys)
    assert code == 2 and "already holds results" in text


# ------------------------------------------------------------------------------------------------ pure CLI behaviour
def test_cli_is_never_interactive(tmp_path):
    assert not any("input(" in p.read_text() for p in SRC.glob("*.py"))
    bids = build_bids(tmp_path / "b", [{"id": "001", "cap": 19, "recordings": ec_eo()}])
    r = subprocess.run([sys.executable, "-m", "openneuro_eeg_triage", "--bids-dir", str(bids), "-o", str(tmp_path / "o"), "--no-plots", "--subjects", "1"],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "qc_summary.xlsx" in r.stdout
    r = subprocess.run([sys.executable, "-m", "openneuro_eeg_triage"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "exactly one of --dataset" in r.stderr


def test_argument_errors(capsys, tmp_path):
    for argv in (["--dataset", "ds000001", "--bids-dir", "x"], ["--dataset", "ds000001", "--stream", "--no-cleanup"],
                 ["--bids-dir", "x", "--force", "--overwrite-all"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2
    capsys.readouterr()
    code, text = cli(["--bids-dir", tmp_path, "--max-bad-channel-ratio", "0.1", "--max-bad-channels", "3"], capsys)
    assert code == 2 and "either --max-bad-channel-ratio or --max-bad-channels" in text
    code, text = cli(["--bids-dir", tmp_path, "--set", "bogus=1"], capsys)
    assert code == 2 and "unknown configuration key" in text
    code, text = cli(["--bids-dir", tmp_path / "missing"], capsys)
    assert code == 2 and "not found" in text


def test_every_documented_flag_exists():
    parser = build_parser()
    flags = {o for a in parser._actions for o in a.option_strings}
    for f in ("--dataset", "--bids-dir", "--subjects", "--sessions", "--tasks", "--limit", "--cleanup", "--no-cleanup", "--line-freq",
              "--plots", "--output-dir", "--resume", "--force", "--overwrite-all", "--stream", "--keep-data", "--max-bad-channel-ratio",
              "--max-bad-channels", "--critical-channels", "--min-sfreq", "--max-sfreq", "--download-method", "--inspect"):
        assert f in flags, f


def test_cli_threshold_flags_reach_the_engine(tmp_path, capsys):
    bids = build_bids(tmp_path / "b", [{"id": "001", "cap": 32, "recordings": [{"task": "rest", "flat": ["F3", "F4", "C3", "C4"]}]}])
    code, _ = cli(["--bids-dir", bids, "-o", tmp_path / "o1", "--no-plots", "-q"], capsys)           # max 3 of 32 -> FAIL
    assert sheet(tmp_path / "o1", "Subjects").loc[0, "label"] == "REJECT"
    code, _ = cli(["--bids-dir", bids, "-o", tmp_path / "o2", "--no-plots", "-q", "--max-bad-channels", "5", "--line-freq", "60"], capsys)
    row = sheet(tmp_path / "o2", "Subjects").loc[0]
    assert row["label"] == "USABLE_NEEDS_INTERPOLATION"
    assert sheet(tmp_path / "o2", "Recordings").loc[0, "line_hz"] == 60 and sheet(tmp_path / "o2", "Recordings").loc[0, "line_freq_source"] == "config"
    code, _ = cli(["--bids-dir", bids, "-o", tmp_path / "o3", "--no-plots", "-q", "--critical-channels", "F3", "F4", "--max-bad-channels", "9"], capsys)
    assert "CRITICAL_CHANNELS_BAD" in sheet(tmp_path / "o3", "Recordings").loc[0, "codes"]


def test_interrupt_keeps_finished_subjects_and_resume_completes(tmp_path, monkeypatch, capsys):
    bids = build_bids(tmp_path / "b", [{"id": f"{i:03d}", "cap": 19, "recordings": [{"task": "rest"}]} for i in range(1, 6)])
    real, calls = pipeline.process_subject, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "process_subject", flaky)
    out = tmp_path / "o"
    code, text = cli(["--bids-dir", bids, "-o", out, "--no-plots"], capsys)
    assert code == 1 and "interrupted" in text and len(sheet(out, "Subjects")) == 2        # workbook written despite Ctrl+C
    monkeypatch.setattr(pipeline, "process_subject", real)
    code, text = cli(["--bids-dir", bids, "-o", out, "--no-plots"], capsys)
    assert code == 0 and text.count("already analysed, skipped") == 2 and len(sheet(out, "Subjects")) == 5


def test_figures_are_written_per_subject(tmp_path, capsys):
    bids = build_bids(tmp_path / "b", [{"id": "001", "cap": 32, "recordings": ec_eo()}])
    code, _ = cli(["--bids-dir", bids, "-o", tmp_path / "o", "-q"], capsys)
    figs = sorted(p.name for p in (tmp_path / "o" / "sub-001" / "figures").glob("*.png"))
    assert code == 0 and figs == ["sub-001_amplitude_heatmap.png", "sub-001_channel_quality_topomap.png", "sub-001_psd_posterior.png"]
    assert all(p.stat().st_size > 5000 for p in (tmp_path / "o" / "sub-001" / "figures").glob("*.png"))


def test_unsupported_montage_dataset_is_reported_not_crashed(tmp_path, capsys):
    egi = [f"E{i}" for i in range(1, 33)]
    bids = build_bids(tmp_path / "b", [{"id": "001", "cap": 32, "recordings": [{"task": "rest", "hw_names": egi}]},
                                       {"id": "002", "cap": 32, "recordings": [{"task": "rest"}]}])
    code, _ = cli(["--bids-dir", bids, "-o", tmp_path / "o", "--no-plots", "-q"], capsys)
    subj = sheet(tmp_path / "o", "Subjects").set_index("subject")
    assert code == 0 and subj.loc["sub-001", "label"] == "REJECT" and "UNSUPPORTED_MONTAGE" in subj.loc["sub-001", "reason_codes"]
    assert subj.loc["sub-002", "label"] in ("USABLE", "USABLE_WITH_FLAGS")
    assert "UNSUPPORTED_MONTAGE" in sheet(tmp_path / "o", "Issues")["code"].tolist()


# ------------------------------------------------------------------------------------------------ fake OpenNeuro, full stack
def mixed_dataset(root: Path):
    subs = [{"id": "001", "cap": 32, "fmt": "edf", "recordings": ec_eo()},
            {"id": "002", "cap": 19, "fmt": "vhdr", "recordings": ec_eo()},
            {"id": "003", "cap": 32, "fmt": "set", "recordings": ec_eo()},
            {"id": "004", "cap": 32, "fmt": "bdf", "recordings": ec_eo()},
            {"id": "005", "cap": 64, "fmt": "fif", "recordings": ec_eo()},
            {"id": "006", "cap": 32, "fmt": "edf", "recordings": ec_eo(line_hz=60, line_amp=8), "sidecar_line": 60}]
    return build_bids(root, subs, "Fake OpenNeuro set")


def test_dataset_inspect_downloads_nothing(fake_openneuro, capsys):
    mixed_dataset(fake_openneuro.root / "ds000100")
    code, text = cli(["--dataset", "ds000100", "--inspect"], capsys)
    assert code == 0 and "Subjects     : 6" in text and "Recordings   : 12" in text and "bdf x2" in text and "Fake OpenNeuro set" in text
    assert fake_openneuro.hits == []


def test_dataset_download_screen_delete_end_to_end(fake_openneuro, tmp_path, capsys):
    mixed_dataset(fake_openneuro.root / "ds000101")
    out, work = tmp_path / "out", tmp_path / "work"
    code, text = cli(["--dataset", "ds000101", "-o", out, "--work-dir", work, "--stream", "--download-method", "https", "--no-plots"], capsys)
    assert code == 0 and "Download–Screen–Delete (single-recording peak disk footprint)" in text
    assert all_files(work) == []                                                        # nothing left on disk
    events = [json.loads(line) for line in (out / "audit_trail.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "downloaded" for e in events) == 12 and sum(e["event"] == "cleaned" for e in events) == 12
    hit_ext = {Path(h).suffix for h in fake_openneuro.hits}
    assert {".edf", ".vhdr", ".vmrk", ".eeg", ".set", ".fdt", ".bdf", ".fif"} <= hit_ext   # every companion file was fetched
    subj = sheet(out, "Subjects").set_index("subject")
    assert len(subj) == 6 and subj.loc["sub-002", "n_eeg_channels"] == 19 and subj.loc["sub-005", "n_eeg_channels"] == 64
    assert all(subj["status"] == "COMPLETED") and subj["data_deleted"].all()
    rec = sheet(out, "Recordings")
    assert set(rec["format"]) == {"edf", "vhdr", "set", "bdf", "fif"} and set(rec[rec["subject"] == "sub-006"]["line_hz"]) == {60}
    full = pd.read_csv(out / "qc_results.csv")
    assert full["inputs_sha256"].str.len().min() > 12
    marker = json.loads((out / ".openneuro_eeg_triage.json").read_text())
    assert marker["dataset_id"] == "ds000101" and marker["snapshot"] == "1.0.2"


def test_keep_data_keeps_files(fake_openneuro, tmp_path, capsys):
    build_bids(fake_openneuro.root / "ds000102", [{"id": "001", "cap": 19, "recordings": [{"task": "rest"}]}])
    work = tmp_path / "work"
    code, _ = cli(["--dataset", "ds000102", "-o", tmp_path / "o", "--work-dir", work, "--keep-data", "--no-plots", "-q", "--download-method", "https"], capsys)
    assert code == 0 and any(p.suffix == ".edf" for p in all_files(work))


def test_failed_downloads_are_partial_and_retried_on_resume(fake_openneuro, tmp_path, capsys):
    root = fake_openneuro.root / "ds000103"
    build_bids(root, [{"id": f"{i:03d}", "cap": 19, "recordings": [{"task": "rest"}]} for i in (1, 2, 3)])
    fake_openneuro.fail_status = 404
    fake_openneuro.fail = {"sub-002/eeg/sub-002_task-rest_eeg.edf"}
    out, work = tmp_path / "out", tmp_path / "work"
    args = ["--dataset", "ds000103", "-o", out, "--work-dir", work, "--no-plots", "--download-method", "https"]
    code, text = cli(args, capsys)
    assert code == 1 and "1 failed" in text and "not finished: sub-002" in text
    subj = sheet(out, "Subjects").set_index("subject")
    assert subj.loc["sub-002", "status"] == "NO_DATA"          # its only recording could not be downloaded
    assert subj.loc["sub-001", "status"] == "COMPLETED" and all_files(work) == []
    fake_openneuro.fail = set()
    code, text = cli(args, capsys)
    assert code == 0 and text.count("already analysed, skipped") == 2
    assert sheet(out, "Subjects").set_index("subject").loc["sub-002", "status"] == "COMPLETED"


def test_stop_after_consecutive_download_failures(fake_openneuro, tmp_path, capsys):
    root = fake_openneuro.root / "ds000104"
    build_bids(root, [{"id": f"{i:03d}", "cap": 19, "recordings": [{"task": "rest"}]} for i in range(1, 7)])
    fake_openneuro.fail_status = 404
    fake_openneuro.fail = {f"sub-{i:03d}/eeg/sub-{i:03d}_task-rest_eeg.edf" for i in range(1, 7)}
    code, text = cli(["--dataset", "ds000104", "-o", tmp_path / "o", "--work-dir", tmp_path / "w", "--no-plots", "--download-method", "https",
                      "--stop-after-failures", "2"], capsys)
    assert code == 1 and "stopped after 2 subjects" in text
    assert len(sheet(tmp_path / "o", "Subjects")) == 2


def test_dataset_error_exit_codes(fake_openneuro, tmp_path, capsys):
    root = fake_openneuro.root / "ds000105"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"0")
    code, text = cli(["--dataset", "ds000105", "--inspect"], capsys)
    assert code == 2 and "no EEG recordings" in text and "anat" in text
    code, text = cli(["--dataset", "ds999999", "--inspect"], capsys)
    assert code == 2 and "not found on OpenNeuro" in text
    build_bids(fake_openneuro.root / "ds000106", [{"id": "001", "cap": 19, "recordings": [{"task": "rest"}]}])
    code, text = cli(["--dataset", "ds000106", "--subjects", "7", "--inspect"], capsys)
    assert code == 2 and "not found in the dataset" in text
    code, text = cli(["--dataset", "ds000106", "--tasks", "nothing", "--inspect"], capsys)
    assert code == 2 and "no recordings match" in text


def test_datasets_cannot_share_an_output_directory(fake_openneuro, tmp_path, capsys):
    for ds in ("ds000107", "ds000108"):
        build_bids(fake_openneuro.root / ds, [{"id": "001", "cap": 19, "recordings": [{"task": "rest"}]}])
    out = tmp_path / "o"
    assert cli(["--dataset", "ds000107", "-o", out, "--work-dir", tmp_path / "w", "--no-plots", "-q", "--download-method", "https"], capsys)[0] == 0
    code, text = cli(["--dataset", "ds000108", "-o", out, "--work-dir", tmp_path / "w", "--no-plots", "-q", "--download-method", "https"], capsys)
    assert code == 2 and "already holds results for openneuro:ds000107" in text
