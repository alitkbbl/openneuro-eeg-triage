"""Synthetic test fixtures: no network, no real data.

* ``synth_uv``            clean or defective multichannel EEG (pink noise + shared component + occipital alpha)
* ``write_*``             EDF, BDF (hand-written 24-bit writer), BrainVision, EEGLAB, FIF
* ``build_bids``          a small BIDS dataset on disk (any cap, any format, any layout)
* ``FakeOpenNeuro``       a local HTTP server that speaks the subset of the GitHub / S3 API the package uses, so the
                          REAL downloader and discovery code run against it
"""
from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mne
import numpy as np
import pytest

from openneuro_eeg_triage.config import QCConfig
from openneuro_eeg_triage.engine import Recording, normalize_channel_name

mne.set_log_level("ERROR")

CAP_19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
CAP_19_LEGACY = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
CAP_32 = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
          "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2"]
CAP_64 = list(mne.channels.make_standard_montage("biosemi64").ch_names)
CAPS = {19: CAP_19, 32: CAP_32, 64: CAP_64}


def cap128() -> list[str]:
    names = [n for n in mne.channels.make_standard_montage("standard_1005").ch_names if n not in ("A1", "A2", "M1", "M2", "T3", "T4", "T5", "T6")]
    keep = [n for n in ("Fz", "Cz", "Pz", "Oz", "O1", "O2", "F3", "F4", "C3", "C4", "Fp1", "Fp2") if n in names]
    rest = [n for n in names if n not in keep][: 128 - len(keep)]
    return keep + rest


# ------------------------------------------------------------------------------------------------ signal
def _pink(rng: np.random.Generator, n: int, fs: float, std: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std() * std


def synth_uv(names: list[str], sfreq: float = 128.0, duration: float = 130.0, seed: int = 0, alpha_amp: float = 25.0,
             line_hz: float | None = None, line_amp: float = 0.0, flat=(), noisy=(), uncorrelated=(), clip=(),
             frozen: tuple[float, float] | None = None, scale: float = 1.0, independent: bool = False) -> np.ndarray:
    """Microvolt data (n_channels, n_samples). Defects are applied by channel name."""
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    common = _pink(rng, n, sfreq, 15.0)
    t = np.arange(n) / sfreq
    data = np.zeros((len(names), n))
    for i, nm in enumerate(names):
        indep = _pink(rng, n, sfreq, 6.0) + rng.standard_normal(n) * 1.5
        x = indep if independent else (rng.uniform(0.8, 1.2) * common + indep)
        if nm in ("O1", "O2", "Oz", "PO3", "PO4", "POz", "P3", "P4"):
            mod = 1 + 0.3 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))
            x = x + alpha_amp * mod * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6))
        if line_hz and line_amp:
            x = x + line_amp * np.sin(2 * np.pi * line_hz * t + rng.uniform(0, 6))
        if nm in flat:
            x = np.zeros(n)
        elif nm in noisy:
            x = _pink(rng, n, sfreq, 250.0)
        elif nm in uncorrelated:
            x = rng.standard_normal(n) * 15.0          # a disconnected electrode: thermal (white) noise only
        if nm in clip:
            lim = 0.4 * x.std()
            x = np.clip(x, -lim, lim)
        data[i] = x
    if frozen:
        s0, length = int(frozen[0] * sfreq), int(frozen[1] * sfreq)
        data[:, s0:s0 + length] = data[:, s0 - 1:s0]
    return data * scale


def make_recording(names: list[str], cfg: QCConfig | None = None, sfreq: float = 128.0, **kw) -> Recording:
    cfg = cfg or QCConfig()
    data = synth_uv(names, sfreq=sfreq, **kw)
    return Recording(ch_names=[normalize_channel_name(n)[0] for n in names], raw_ch_names=list(names), sfreq=sfreq,
                     data_uv=data, cfg=cfg)


def to_raw(names: list[str], data_uv: np.ndarray, sfreq: float) -> mne.io.RawArray:
    info = mne.create_info(list(names), sfreq, "eeg")
    return mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")


# ------------------------------------------------------------------------------------------------ writers
def write_edf(path: Path, names, data_uv, sfreq) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(path), to_raw(names, data_uv, sfreq), fmt="edf", overwrite=True, verbose="ERROR")
    return [path]


def write_bdf(path: Path, names, data_uv, sfreq) -> list[Path]:
    """Minimal 24-bit BioSemi BDF writer (1-second data records)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ns, fs = len(names), int(sfreq)
    n_rec = data_uv.shape[1] // fs
    phys = 4096.0
    dig = np.clip(np.rint(data_uv[:, : n_rec * fs] / phys * 8388607), -8388608, 8388607).astype("<i4")
    f = lambda s, n: s.encode("ascii")[:n].ljust(n)                                         # noqa: E731
    head = b"\xffBIOSEMI" + f("X", 80) + f("Startdate 01-JAN-2020", 80) + f("01.01.20", 8) + f("00.00.00", 8)
    head += f(str(256 * (ns + 1)), 8) + f("24BIT", 44) + f(str(n_rec), 8) + f("1", 8) + f(str(ns), 4)
    for field_len, value in ((16, lambda i: names[i]), (80, lambda i: "Active Electrode"), (8, lambda i: "uV"),
                             (8, lambda i: str(-phys)), (8, lambda i: str(phys)), (8, lambda i: "-8388608"),
                             (8, lambda i: "8388607"), (80, lambda i: "none"), (8, lambda i: str(fs)), (32, lambda i: "")):
        for i in range(ns):
            head += f(value(i), field_len)
    body = bytearray()
    for r in range(n_rec):
        for i in range(ns):
            seg = dig[i, r * fs:(r + 1) * fs]
            body += seg.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    path.write_bytes(bytes(head) + bytes(body))
    return [path]


def write_vhdr(path: Path, names, data_uv, sfreq) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(path), to_raw(names, data_uv, sfreq), fmt="brainvision", overwrite=True, verbose="ERROR")
    return [path, path.with_suffix(".vmrk"), path.with_suffix(".eeg")]


def write_set(path: Path, names, data_uv, sfreq, fdt: bool = True) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(path), to_raw(names, data_uv, sfreq), fmt="eeglab", overwrite=True, verbose="ERROR")
    out = [path]
    if fdt:                                       # companion presence is what the streamer must handle
        fp = path.with_suffix(".fdt")
        fp.write_bytes(b"\x00" * 64)
        out.append(fp)
    return out


def write_fif(path: Path, names, data_uv, sfreq) -> list[Path]:
    import warnings
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_raw(names, data_uv, sfreq).save(str(path), overwrite=True, verbose="ERROR")
    return [path]


WRITERS = {"edf": write_edf, "bdf": write_bdf, "vhdr": write_vhdr, "set": write_set, "fif": write_fif}


# ------------------------------------------------------------------------------------------------ BIDS builder
def build_bids(root: Path, subjects: list[dict], dataset_name: str = "Synthetic EEG", with_participants: bool = True) -> Path:
    """Write a BIDS dataset. Each subject dict::

        {"id": "001", "cap": 19|32|64|list_of_names, "fmt": "edf", "sessions": ["1"] or [None],
         "recordings": [{"task": "EyesClosed", "acq": "pre", "run": None, **synth kwargs}, ...],
         "line_hz": 50, "sidecar_line": 50 | None, "age": 30, "channels_tsv": True}
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": dataset_name, "BIDSVersion": "1.8.0", "DatasetType": "raw"}))
    rows = ["participant_id\tage\tsex\thandedness"]
    for s in subjects:
        names = CAPS[s["cap"]] if isinstance(s.get("cap", 19), int) else list(s["cap"])
        fmt = s.get("fmt", "edf")
        for ses in s.get("sessions", [None]):
            for k, r in enumerate(s["recordings"]):
                r = dict(r)
                task, acq, run = r.pop("task", "rest"), r.pop("acq", None), r.pop("run", None)
                sfreq = r.pop("sfreq", s.get("sfreq", 128.0))
                hw_names = r.pop("hw_names", None) or names
                stem = f"sub-{s['id']}" + (f"_ses-{ses}" if ses else "") + f"_task-{task}" + (f"_acq-{acq}" if acq else "") + (f"_run-{run}" if run else "")
                d = root / f"sub-{s['id']}" / (f"ses-{ses}" if ses else "") / "eeg"
                data = synth_uv(names, sfreq=sfreq, seed=int(s["id"]) * 100 + k, **{**s.get("signal", {}), **r})
                WRITERS[fmt](d / f"{stem}_eeg.{fmt}", hw_names, data, sfreq)
                side = {"SamplingFrequency": sfreq, "EEGChannelCount": len(names), "TaskName": task}
                if s.get("sidecar_line") is not None:
                    side["PowerLineFrequency"] = s["sidecar_line"]
                (d / f"{stem}_eeg.json").write_text(json.dumps(side))
                if s.get("channels_tsv", True):
                    (d / f"{stem}_channels.tsv").write_text("name\ttype\tunits\n" + "".join(f"{n}\tEEG\tuV\n" for n in hw_names))
        rows.append(f"sub-{s['id']}\t{s.get('age', 30)}\t{s.get('sex', 'F')}\tR")
    if with_participants:
        (root / "participants.tsv").write_text("\n".join(rows) + "\n")
    return root


def ec_eo(**ec_kw) -> list[dict]:
    """An Eyes-Closed (strong alpha) and an Eyes-Open (weak alpha) recording."""
    return [{"task": "EyesClosed", **ec_kw}, {"task": "EyesOpen", "alpha_amp": 3.0, **{k: v for k, v in ec_kw.items() if k != "alpha_amp"}}]


# ------------------------------------------------------------------------------------------------ fake OpenNeuro
class FakeOpenNeuro:
    """Local server for the GitHub tree/tags API, raw files and the S3 bucket (GET + ListObjectsV2).

    Layout: ``root/<dsid>/...`` holds the datasets. Knobs: ``truncate`` (GitHub tree says truncated),
    ``github_down`` (API returns 404/503), ``fail`` (set of relative paths answering 500), ``hits`` (S3 file requests).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.truncate = False
        self.github_down = False
        self.fail: set[str] = set()
        self.fail_status = 500
        self.hits: list[str] = []
        self.tags = ["1.0.0", "1.0.2"]
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):                  # silence
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                path, q = urllib.parse.unquote(u.path), urllib.parse.parse_qs(u.query)
                try:
                    if path.startswith("/api/"):
                        return self.api(path[5:], q)
                    if path.startswith("/raw/"):
                        parts = path[5:].split("/", 3)          # OpenNeuroDatasets/<ds>/<ref>/<rel>
                        return self.file(Path(outer.root, parts[1], parts[3]))
                    if path.startswith("/s3"):
                        rel = path[4:].lstrip("/")
                        if not rel:
                            return self.listing(q)
                        outer.hits.append(rel)
                        if rel.split("/", 1)[-1] in outer.fail:
                            return self._send(outer.fail_status, b"boom", "text/plain")
                        return self.file(Path(outer.root, rel))
                    self._send(404, b"{}")
                except BrokenPipeError:
                    pass

            def file(self, p: Path):
                if p.is_file():
                    self._send(200, p.read_bytes(), "application/octet-stream")
                else:
                    self._send(404, b"not found", "text/plain")

            def api(self, p: str, q):
                if outer.github_down:
                    return self._send(503, b"{}")
                parts = p.split("/")                              # repos/OpenNeuroDatasets/<ds>/...
                ds = parts[2]
                base = Path(outer.root, ds)
                if not base.is_dir():
                    return self._send(404, b'{"message":"Not Found"}')
                if parts[3] == "tags":
                    return self._send(200, json.dumps([{"name": t} for t in outer.tags]).encode())
                if parts[3] == "git" and parts[4] == "trees":
                    ref = "/".join(parts[5:])
                    if "recursive" in q:
                        if outer.truncate:
                            return self._send(200, json.dumps({"truncated": True, "tree": []}).encode())
                        tree = [{"path": f.relative_to(base).as_posix(), "type": "blob", "size": f.stat().st_size}
                                for f in sorted(base.rglob("*")) if f.is_file()]
                        return self._send(200, json.dumps({"truncated": False, "tree": tree}).encode())
                    rel = "" if ref in outer.tags or ref == "HEAD" else ref[2:]      # sha = "t:<path>"
                    d = base / rel
                    tree = []
                    for c in sorted(d.iterdir()):
                        r = c.relative_to(base).as_posix()
                        tree.append({"path": c.name, "type": "tree" if c.is_dir() else "blob", "sha": f"t:{r}",
                                     "size": c.stat().st_size if c.is_file() else None})
                    return self._send(200, json.dumps({"truncated": False, "tree": tree}).encode())
                self._send(404, b"{}")

            def listing(self, q):
                prefix = q.get("prefix", [""])[0]
                files = [f for f in sorted(Path(outer.root).rglob("*")) if f.is_file()
                         and f.relative_to(outer.root).as_posix().startswith(prefix)]
                xml = ['<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
                       "<IsTruncated>false</IsTruncated>"]
                for f in files:
                    xml.append(f"<Contents><Key>{f.relative_to(outer.root).as_posix()}</Key><Size>{f.stat().st_size}</Size></Contents>")
                xml.append("</ListBucketResult>")
                self._send(200, "".join(xml).encode(), "application/xml")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    @property
    def env(self) -> dict[str, str]:
        b = f"http://127.0.0.1:{self.port}"
        return {"OPENNEURO_EEG_TRIAGE_GITHUB_API": f"{b}/api", "OPENNEURO_EEG_TRIAGE_GITHUB_RAW": f"{b}/raw",
                "OPENNEURO_EEG_TRIAGE_S3_BASE": f"{b}/s3"}


@pytest.fixture()
def fake_openneuro(tmp_path, monkeypatch):
    srv = FakeOpenNeuro(tmp_path / "server_root").start()
    for k, v in srv.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GITHUB_TOKEN", "")
    yield srv
    srv.stop()


@pytest.fixture()
def cfg() -> QCConfig:
    return QCConfig()


def have_soffice() -> bool:
    import shutil
    return shutil.which("soffice") is not None or shutil.which("libreoffice") is not None
