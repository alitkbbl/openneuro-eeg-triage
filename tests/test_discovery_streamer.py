"""Discovery, selection and Download–Screen–Delete streaming (real HTTP code against a local fake OpenNeuro)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from conftest import build_bids

from openneuro_eeg_triage import pipeline
from openneuro_eeg_triage.config import QCConfig
from openneuro_eeg_triage.discovery import (DatasetNotFoundError, DiscoveryError, NoEEGDataError, SelectionError,
                                            discover_from_paths, discover_local, discover_openneuro, parse_bids_name,
                                            parse_subject_tokens, resolve_subjects, select_recordings)
from openneuro_eeg_triage.pipeline import AuditTrail, RunOptions, screen_recording
from openneuro_eeg_triage.streamer import (BackendError, DataladBackend, DownloadError, HttpsBackend, LocalBackend,
                                           make_backend, parse_vhdr_references)
from openneuro_eeg_triage.utils import Logger

SHORT = {"duration": 4.0}            # discovery/streaming tests do not need analysable recordings


def one(sid, fmt="edf", **kw):
    return {"id": sid, "cap": 32, "fmt": fmt, "recordings": [{"task": "rest", **SHORT}], **kw}


def all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# ------------------------------------------------------------------------------------------------ BIDS name parsing
def test_parse_bids_name():
    info = parse_bids_name("sub-01_ses-2_task-rest_acq-pre_run-3_eeg.vhdr")
    assert info["entities"] == {"sub": "01", "ses": "2", "task": "rest", "acq": "pre", "run": "3"}
    assert info["suffix"] == "eeg" and info["ext"] == ".vhdr"
    assert parse_bids_name("README") is None and parse_bids_name("notbids.txt") is None


# ------------------------------------------------------------------------------------------------ local discovery
def test_local_discovery_all_formats_and_companions(tmp_path):
    subs = [one("001", "edf"), one("002", "bdf"), one("003", "vhdr"), one("004", "set"), one("005", "fif")]
    build_bids(tmp_path / "bids", subs, "Formats")
    info = discover_local(tmp_path / "bids")
    assert info.title == "Formats" and info.source == "local" and len(info.subjects) == 5
    assert info.formats == {"edf": 1, "bdf": 1, "vhdr": 1, "set": 1, "fif": 1}
    by = {r.subject: r for r in info.recordings}
    assert sorted(Path(c).suffix for c in by["sub-003"].companions) == [".eeg", ".vmrk"]       # BrainVision triple
    assert [Path(c).suffix for c in by["sub-004"].companions] == [".fdt"]                      # EEGLAB .set + .fdt
    assert by["sub-001"].companions == []
    for r in info.recordings:
        assert set(r.sidecars) == {"json", "channels"}
    assert info.participants["sub-001"]["age"] == "30"
    text = "\n".join(info.summary_lines())
    assert "Subjects" in text and "Formats" in text and "5" in text


def test_sessions_acq_run_and_single_session_layouts(tmp_path):
    two = {"id": "001", "cap": 32, "sessions": ["1", "2"], "recordings": [{"task": "EyesClosed", "acq": "pre", **SHORT}, {"task": "EyesOpen", "acq": "post", "run": "2", **SHORT}]}
    build_bids(tmp_path / "b", [two, one("002")])
    info = discover_local(tmp_path / "b")
    assert info.sessions == ["ses-1", "ses-2"] and info.tasks == ["EyesClosed", "EyesOpen", "rest"]
    r = next(x for x in info.recordings if x.run == "2")
    assert (r.subject, r.session, r.task, r.acq, r.run) == ("sub-001", "ses-1", "EyesOpen", "post", "2")
    assert next(x for x in info.recordings if x.subject == "sub-002").session is None
    assert r.rec_id == "sub-001_ses-1_task-EyesOpen_acq-post_run-2"


def test_derivatives_and_sourcedata_are_ignored(tmp_path):
    root = build_bids(tmp_path / "b", [one("001")])
    for top in ("derivatives/pipe", "sourcedata"):
        d = root / top / "sub-001" / "eeg"
        d.mkdir(parents=True)
        (d / "sub-001_task-x_eeg.edf").write_bytes(b"0" * 10)
    assert len(discover_local(root).recordings) == 1


def test_bids_inheritance_of_sidecars(tmp_path):
    root = build_bids(tmp_path / "b", [one("001")])
    for f in (root / "sub-001" / "eeg").glob("*_eeg.json"):
        f.unlink()
    (root / "task-rest_eeg.json").write_text(json.dumps({"PowerLineFrequency": 60}))
    r = discover_local(root).recordings[0]
    assert r.sidecars["json"] == "task-rest_eeg.json"


def test_mri_only_dataset_is_reported_cleanly(tmp_path):
    root = tmp_path / "mri"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"0")
    with pytest.raises(NoEEGDataError, match="anat"):
        discover_local(root)
    with pytest.raises(DatasetNotFoundError):
        discover_local(tmp_path / "does_not_exist")


# ------------------------------------------------------------------------------------------------ selection
def test_subject_token_parsing_and_resolution():
    avail = ["sub-001", "sub-002", "sub-003", "sub-010", "sub-PD01"]
    assert parse_subject_tokens(["1", "to", "3"]) == [("range", 1, 3)]
    assert resolve_subjects(["1", "to", "3"], avail) == ["sub-001", "sub-002", "sub-003"]
    assert resolve_subjects(["10", "sub-002", "pd01"], avail) == ["sub-002", "sub-010", "sub-PD01"]
    assert resolve_subjects(["1-2,10"], avail) == ["sub-001", "sub-002", "sub-010"]
    assert resolve_subjects(["3", "1", "3"], avail) == ["sub-001", "sub-003"]            # duplicates ignored
    with pytest.raises(SelectionError, match="99"):
        resolve_subjects(["1", "99"], avail)
    with pytest.raises(SelectionError):
        resolve_subjects(["a", "to", "3"], avail)


def test_select_recordings_filters_limit_and_empty(tmp_path):
    subs = [{"id": f"{i:03d}", "cap": 32, "sessions": ["1", "2"], "recordings": [{"task": "EyesClosed", **SHORT}, {"task": "EyesOpen", **SHORT}]} for i in range(1, 6)]
    info = discover_local(build_bids(tmp_path / "b", subs))
    assert len(select_recordings(info)) == 5
    sel = select_recordings(info, sessions=["2"], tasks=["EyesOpen"])
    assert all(len(v) == 1 and v[0].session == "ses-2" and v[0].task == "EyesOpen" for v in sel.values())
    assert list(select_recordings(info, limit=2)) == ["sub-001", "sub-002"]
    assert list(select_recordings(info, subjects=["4", "5"], limit=1)) == ["sub-004"]
    with pytest.raises(SelectionError, match="no recordings match"):
        select_recordings(info, tasks=["nothing"])
    with pytest.raises(SelectionError):
        select_recordings(info, limit=0)


def test_discover_from_paths_is_pure():
    specs, mods = discover_from_paths([("sub-1/eeg/sub-1_task-a_eeg.edf", 5), ("sub-1/anat/sub-1_T1w.nii.gz", 9)])
    assert len(specs) == 1 and specs[0].size == 5 and mods == {"eeg": 1, "anat": 1}


# ------------------------------------------------------------------------------------------------ OpenNeuro discovery
def test_openneuro_discovery_via_github_tree(fake_openneuro):
    build_bids(fake_openneuro.root / "ds000001", [one("001"), one("002", "vhdr")], "My dataset")
    info = discover_openneuro("ds000001")
    assert info.source == "openneuro" and info.snapshot == "1.0.2" and "GitHub tree" in info.discovery_method
    assert info.title == "My dataset" and len(info.recordings) == 2 and info.participants["sub-002"]["sex"] == "F"
    assert fake_openneuro.hits == []                                   # discovery downloaded no data file
    assert discover_openneuro("DS000001", snapshot="1.0.0").snapshot == "1.0.0"


def test_openneuro_truncated_tree_falls_back_to_directory_walk(fake_openneuro):
    build_bids(fake_openneuro.root / "ds000002", [{"id": "001", "cap": 32, "sessions": ["1"], "recordings": [{"task": "rest", **SHORT}]}, one("002")])
    fake_openneuro.truncate = True
    info = discover_openneuro("ds000002")
    assert len(info.recordings) == 2 and "directory walk" in info.discovery_method


def test_openneuro_s3_fallback_when_github_is_down(fake_openneuro):
    build_bids(fake_openneuro.root / "ds000003", [one("001")])
    fake_openneuro.github_down = True
    info = discover_openneuro("ds000003")
    assert "S3" in info.discovery_method and len(info.recordings) == 1 and any("newest snapshot" in w for w in info.warnings)
    with pytest.raises(DiscoveryError, match="cannot inspect snapshot"):
        discover_openneuro("ds000003", snapshot="1.0.0")               # a pinned version cannot be verified without GitHub


def test_openneuro_error_cases(fake_openneuro):
    with pytest.raises(DatasetNotFoundError, match="not found"):
        discover_openneuro("ds999999")
    with pytest.raises(DatasetNotFoundError, match="not an OpenNeuro dataset id"):
        discover_openneuro("hello")
    root = fake_openneuro.root / "ds000004"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"0")
    (root / "dataset_description.json").write_text("{}")
    with pytest.raises(NoEEGDataError, match="no EEG recordings"):
        discover_openneuro("ds000004")
    build_bids(fake_openneuro.root / "ds000005", [one("001")])
    with pytest.raises(DiscoveryError, match="snapshot '9.9' not found"):
        discover_openneuro("ds000005", snapshot="9.9")


# ------------------------------------------------------------------------------------------------ streaming backends
def https(fake, ds, tmp_path):
    info = discover_openneuro(ds)
    return info, HttpsBackend(ds, tmp_path / "work")


def test_https_backend_edf_sidecars_first_then_data_then_cleanup(fake_openneuro, tmp_path):
    build_bids(fake_openneuro.root / "ds000010", [one("001")])
    info, be = https(fake_openneuro, "ds000010", tmp_path)
    spec = info.recordings[0]
    s = be.new_session()
    be.fetch_sidecars(spec, s)
    assert set(s.sidecars) == {"json", "channels"} and not any(f.suffix == ".edf" for f in s.files)
    be.fetch_data(spec, s)
    assert s.main.exists() and s.main.suffix == ".edf" and s.bytes_downloaded > 0
    order = fake_openneuro.hits
    assert order.index(next(h for h in order if h.endswith("_channels.tsv"))) < order.index(next(h for h in order if h.endswith(".edf")))
    res = be.cleanup(s)
    assert res["deleted"] >= 3 and not res["errors"]
    assert all_files(tmp_path / "work") == [] and not any((tmp_path / "work").iterdir())       # files AND empty folders gone


def test_https_backend_brainvision_triple_and_header_references(fake_openneuro, tmp_path):
    build_bids(fake_openneuro.root / "ds000011", [one("001", "vhdr")])
    info, be = https(fake_openneuro, "ds000011", tmp_path)
    s = be.new_session()
    be.fetch_sidecars(info.recordings[0], s)
    be.fetch_data(info.recordings[0], s)
    names = sorted(p.suffix for p in s.files if p.suffix in (".vhdr", ".vmrk", ".eeg"))
    assert names == [".eeg", ".vhdr", ".vmrk"]
    refs = parse_vhdr_references(s.main)
    assert refs["datafile"].endswith(".eeg") and refs["markerfile"].endswith(".vmrk")
    be.cleanup(s)
    assert all_files(tmp_path / "work") == []


def test_https_backend_follows_vhdr_references_that_differ_from_bids_names(fake_openneuro, tmp_path):
    build_bids(fake_openneuro.root / "ds000012", [one("001", "vhdr")])
    d = fake_openneuro.root / "ds000012" / "sub-001" / "eeg"
    vhdr = next(d.glob("*.vhdr"))
    stem = vhdr.stem
    (d / f"{stem}.eeg").rename(d / "recording.dat")                    # data file named differently from the BIDS stem
    vhdr.write_text(vhdr.read_text().replace(f"{stem}.eeg", "recording.dat"))
    info = discover_openneuro("ds000012")
    be = HttpsBackend("ds000012", tmp_path / "work")
    s = be.new_session()
    be.fetch_data(info.recordings[0], s)
    assert (tmp_path / "work" / "sub-001" / "eeg" / "recording.dat").exists()
    be.cleanup(s)
    assert all_files(tmp_path / "work") == []


def test_https_backend_eeglab_set_with_fdt(fake_openneuro, tmp_path):
    build_bids(fake_openneuro.root / "ds000013", [one("001", "set")])
    info, be = https(fake_openneuro, "ds000013", tmp_path)
    s = be.new_session()
    be.fetch_data(info.recordings[0], s)
    assert sorted(p.suffix for p in s.files) == [".fdt", ".set"]
    be.cleanup(s)
    assert all_files(tmp_path / "work") == []


def test_failed_download_cleans_up_partial_files_and_raises(fake_openneuro, tmp_path):
    build_bids(fake_openneuro.root / "ds000014", [one("001", "vhdr")])
    info, be = https(fake_openneuro, "ds000014", tmp_path)
    be.http.retries = 1
    be.http.backoff = 0.01
    spec = info.recordings[0]
    fake_openneuro.fail = {spec.path.replace(".vhdr", ".eeg")}          # the big companion fails
    s = be.new_session()
    with pytest.raises(DownloadError, match="companion"):
        be.fetch_data(spec, s)
    assert s.main.exists()                                              # the .vhdr had arrived ...
    be.cleanup(s)
    assert all_files(tmp_path / "work") == []                           # ... and is removed anyway


def test_local_backend_never_deletes(tmp_path):
    root = build_bids(tmp_path / "b", [one("001", "vhdr")])
    info = discover_local(root)
    be = LocalBackend(root)
    s = be.new_session()
    be.fetch_sidecars(info.recordings[0], s)
    be.fetch_data(info.recordings[0], s)
    before = sorted(all_files(root))
    assert be.deletes_data is False and be.cleanup(s)["kept"] is True
    assert sorted(all_files(root)) == before
    (root / "sub-001" / "eeg" / next(p.name for p in (root / "sub-001" / "eeg").glob("*.eeg"))).unlink()
    with pytest.raises(DownloadError, match="missing"):
        be.fetch_data(info.recordings[0], be.new_session())


def test_datalad_backend_commands(tmp_path):
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="abcdef1234567890\n", stderr="")

    info = discover_from_paths([("sub-1/eeg/sub-1_task-a_eeg.vhdr", 1), ("sub-1/eeg/sub-1_task-a_eeg.vmrk", 1),
                                ("sub-1/eeg/sub-1_task-a_eeg.eeg", 1), ("sub-1/eeg/sub-1_task-a_channels.tsv", 1)])[0][0]
    be = DataladBackend("ds000001", tmp_path / "repo", tag="1.0.3", runner=runner)
    meta = be.prepare()
    assert meta == {"method": "datalad", "tag": "1.0.3", "commit": "abcdef123456"}
    assert calls[0][:2] == ["datalad", "install"] and ["git", "-C", str(tmp_path / "repo"), "checkout", "-q", "1.0.3"] in calls
    s = be.new_session()
    be.fetch_sidecars(info, s)
    be.fetch_data(info, s)
    got = [c for c in calls if c[:2] == ["datalad", "get"]]
    assert len(got) == 2 and all(str(tmp_path / "repo" / "sub-1" / "eeg") in " ".join(c) for c in got)
    assert any(".vmrk" in a for a in got[1]) and any(".eeg" in a for a in got[1])
    res = be.cleanup(s)
    drop = calls[-1]
    assert drop[:2] == ["datalad", "drop"] and "--what" in drop and res["errors"] == []


def test_backend_selection(tmp_path, monkeypatch):
    root = build_bids(tmp_path / "b", [one("001")])
    assert isinstance(make_backend("auto", discover_local(root), tmp_path / "w", tmp_path / "r"), LocalBackend)
    monkeypatch.setattr(DataladBackend, "available", staticmethod(lambda: False))
    from openneuro_eeg_triage.discovery import DatasetInfo
    info = DatasetInfo("openneuro", "ds000001", "openneuro:ds000001", "", {}, "1.0.3", [], {})
    assert isinstance(make_backend("auto", info, tmp_path / "w", tmp_path / "r"), HttpsBackend)
    with pytest.raises(BackendError, match="datalad"):
        make_backend("datalad", info, tmp_path / "w", tmp_path / "r")
    monkeypatch.setattr(DataladBackend, "available", staticmethod(lambda: True))
    assert isinstance(make_backend("auto", info, tmp_path / "w", tmp_path / "r"), DataladBackend)


# ------------------------------------------------------------------------------------------------ Download–Screen–Delete
def opts(tmp_path, **kw):
    return RunOptions(output_dir=tmp_path / "out", work_dir=tmp_path / "work", repo_dir=tmp_path / "repo", plots=False, **kw)


def test_screen_recording_persists_then_deletes(fake_openneuro, tmp_path):
    build_bids(fake_openneuro.root / "ds000020", [{"id": "001", "cap": 32, "fmt": "vhdr", "recordings": [{"task": "rest"}]}])
    info, be = https(fake_openneuro, "ds000020", tmp_path)
    o = opts(tmp_path)
    record, cl = screen_recording(info.recordings[0], be, QCConfig(), o, AuditTrail(tmp_path / "out" / "audit.jsonl"), Logger(quiet=True))
    assert record["status"] == "OK" and record["inputs"] and all(len(i["sha256"]) == 64 for i in record["inputs"])
    assert (tmp_path / "out" / "sub-001" / "recordings" / f"{record['rec_id']}.json").exists()    # persisted ...
    assert cl["deleted"] >= 4 and all_files(tmp_path / "work") == []                              # ... then deleted
    events = [json.loads(line)["event"] for line in (tmp_path / "out" / "audit.jsonl").read_text().splitlines()]
    assert events == ["recording_start", "downloaded", "screened", "cleaned"]


def test_cleanup_happens_even_when_the_analysis_crashes(fake_openneuro, tmp_path, monkeypatch):
    build_bids(fake_openneuro.root / "ds000021", [one("001")])
    info, be = https(fake_openneuro, "ds000021", tmp_path)

    def boom(*a, **k):
        raise RuntimeError("corrupt file")

    monkeypatch.setattr(pipeline, "load_recording", boom)
    record, cl = screen_recording(info.recordings[0], be, QCConfig(), opts(tmp_path), AuditTrail(tmp_path / "a.jsonl"), Logger(quiet=True))
    assert record["status"] == "ERROR" and "corrupt file" in record["error"]
    assert all_files(tmp_path / "work") == [] and cl["deleted"] > 0
    cached = json.loads((tmp_path / "out" / "sub-001" / "recordings" / f"{record['rec_id']}.json").read_text())
    assert cached["status"] == "ERROR"                                  # persisted before the finally block ran


def test_download_failure_is_missing_not_persisted_and_clean(fake_openneuro, tmp_path):
    build_bids(fake_openneuro.root / "ds000022", [one("001")])
    info, be = https(fake_openneuro, "ds000022", tmp_path)
    be.http.retries, be.http.backoff = 0, 0.01
    fake_openneuro.fail = {info.recordings[0].path}
    record, _ = screen_recording(info.recordings[0], be, QCConfig(), opts(tmp_path), AuditTrail(tmp_path / "a.jsonl"), Logger(quiet=True))
    assert record["status"] == "MISSING" and record["error_code"] == "DOWNLOAD_FAILED"
    assert not (tmp_path / "out" / "sub-001" / "recordings").exists() and all_files(tmp_path / "work") == []


def test_non_standard_montage_is_rejected_before_the_big_file_is_downloaded(fake_openneuro, tmp_path):
    egi = [f"E{i}" for i in range(1, 33)]
    build_bids(fake_openneuro.root / "ds000023", [{"id": "001", "cap": 32, "recordings": [{"task": "rest", "hw_names": egi}]}])
    info, be = https(fake_openneuro, "ds000023", tmp_path)
    record, _ = screen_recording(info.recordings[0], be, QCConfig(), opts(tmp_path), AuditTrail(tmp_path / "a.jsonl"), Logger(quiet=True))
    assert record["status"] == "UNSUPPORTED" and record["layers"]["header"]["issues"][0]["code"] == "UNSUPPORTED_MONTAGE"
    assert not any(h.endswith(".edf") for h in fake_openneuro.hits)     # only sidecars were fetched
    assert all_files(tmp_path / "work") == []
