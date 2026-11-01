"""Command line interface. 100 % non-interactive: every choice is a flag, the tool never asks a question.

Exit codes: 0 finished, 1 technical problem (network, unreadable output, interrupted, subjects not finished),
2 invalid input (bad flags, dataset not found, no EEG data, empty selection).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, QCConfig, QC_LOGIC_VERSION, parse_override
from .discovery import (DatasetNotFoundError, DiscoveryError, NoEEGDataError, SelectionError, discover_local,
                        discover_openneuro, select_recordings)
from .net import Endpoints, Http
from .pipeline import BatchResult, OutputDirError, RunOptions, run_batch
from .streamer import BackendError, make_backend
from .utils import Logger, human_bytes, human_seconds

DESCRIPTION = """\
Automated EEG quality triage for OpenNeuro / BIDS datasets (standard 10-20, 10-10 and 10-05 montages).

Download–Screen–Delete (single-recording peak disk footprint): every recording is fetched with its companion
files, screened by a 5-layer QC engine in memory, its metrics are saved, and the raw files are deleted before the
next recording. Results are cumulative: running other subjects later extends and re-ranks the same workbook.
"""

EPILOG = """\
examples:
  openneuro-eeg-triage --dataset ds005385 --inspect                      what is in the dataset (no EEG download)
  openneuro-eeg-triage --dataset ds005385 --subjects 1 to 20 --stream    screen subjects 1-20, delete after each
  openneuro-eeg-triage --dataset ds004504 --limit 10 --line-freq 50      first 10 subjects, mains frequency fixed
  openneuro-eeg-triage --bids-dir /data/my_bids --tasks rest --max-bad-channels 4
  openneuro-eeg-triage --dataset ds005385 --subjects 21 to 30            later: extends and re-ranks the same results

exit codes: 0 finished | 1 technical problem or unfinished subjects | 2 invalid input / no EEG data
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openneuro-eeg-triage", description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__} (QC logic {QC_LOGIC_VERSION})")

    g = p.add_argument_group("source (exactly one is required)")
    g.add_argument("--dataset", metavar="DSID", help="OpenNeuro dataset id, e.g. ds005385")
    g.add_argument("--bids-dir", metavar="DIR", help="local BIDS directory (files are read in place, never deleted)")
    g.add_argument("--snapshot", metavar="TAG", help="OpenNeuro snapshot tag to inspect (default: the newest)")

    g = p.add_argument_group("selection")
    g.add_argument("--subjects", nargs="+", metavar="S",
                   help="subject numbers or labels: 28 | 1 2 3 | 1 to 20 | 1-20 | 5,9 | sub-PD01 (default: all)")
    g.add_argument("--sessions", nargs="+", metavar="S", help="sessions, e.g. 1 2 or ses-1")
    g.add_argument("--tasks", nargs="+", metavar="T", help="tasks, e.g. rest EyesClosed")
    g.add_argument("--limit", type=int, metavar="N", help="screen only the first N subjects after the filters")
    g.add_argument("--inspect", action="store_true",
                   help="pre-flight only: print the dataset summary and the selection, download nothing, exit")

    g = p.add_argument_group("output and run control")
    g.add_argument("--output-dir", "-o", default="qc_output", metavar="DIR", help="results directory (default: qc_output)")
    g.add_argument("--work-dir", metavar="DIR", help="temporary directory for downloaded recordings (default: ./openneuro_eeg_triage_work)")
    g.add_argument("--repo-dir", metavar="DIR", help="DataLad clone location (default: <work-dir>/<dataset>_repo)")
    g.add_argument("--download-method", choices=["auto", "https", "datalad"], default="auto",
                   help="auto: DataLad if installed, otherwise HTTPS/S3 (default)")
    g.add_argument("--stream", action="store_true", help="Download–Screen–Delete mode (default for --dataset)")
    g.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=None,
                   help="delete the downloaded files of each recording right after it is screened (default: on for --dataset)")
    g.add_argument("--keep-data", action="store_true", help="same as --no-cleanup: keep the downloaded files")
    g.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="write diagnostic figures (default: on)")
    g.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip subjects/recordings already screened with the same thresholds (default: on)")
    g.add_argument("--force", action="store_true", help="re-screen the selected subjects even if results exist (others are kept and merged)")
    g.add_argument("--overwrite-all", action="store_true", help="delete ALL previous results in --output-dir first")
    g.add_argument("--no-hash", action="store_true", help="do not compute SHA-256 of the input files")
    g.add_argument("--stop-after-failures", type=int, default=3, metavar="N",
                   help="stop after N subjects in a row whose download failed (default 3, 0 = never)")
    g.add_argument("--summary-every", type=int, default=25, metavar="N", help="refresh the Excel/CSV summary every N subjects (default 25)")
    g.add_argument("--quiet", "-q", action="store_true", help="only warnings and errors")
    g.add_argument("--debug", action="store_true", help="show tracebacks and stop at the first problem")

    g = p.add_argument_group("QC thresholds (all have defaults; see README)")
    g.add_argument("--max-bad-channel-ratio", type=float, metavar="R",
                   help="max interpolatable channels = max(1, floor(R x n_eeg_channels)); default 0.10")
    g.add_argument("--max-bad-channels", type=int, metavar="N", help="exact maximum number of bad channels (overrides the ratio)")
    g.add_argument("--critical-channels", nargs="+", metavar="CH", help="critical channels (default: Cz Fz Pz O1 O2)")
    g.add_argument("--line-freq", choices=["auto", "50", "60"], help="mains frequency; auto = sidecar PowerLineFrequency, then spectrum")
    g.add_argument("--min-sfreq", type=float, metavar="HZ", help="minimum sampling rate, below it FAIL (default 128)")
    g.add_argument("--max-sfreq", type=float, metavar="HZ", help="maximum sampling rate, above it FLAG (default: none)")
    g.add_argument("--min-duration", type=float, metavar="S", help="recordings shorter than this FAIL (default 60 s)")
    g.add_argument("--min-eeg-channels", type=int, metavar="N", help="fewer EEG channels FAIL (default 16)")
    g.add_argument("--closed-tasks", nargs="+", metavar="LABEL", help="task/acq labels that mean Eyes Closed (extra to the inference)")
    g.add_argument("--open-tasks", nargs="+", metavar="LABEL", help="task/acq labels that mean Eyes Open")
    g.add_argument("--config", metavar="FILE.json", help="JSON file with any QCConfig keys")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override any QCConfig key (VALUE is JSON), repeatable, e.g. --set noisy_ratio=5")
    return p


def build_config(args: argparse.Namespace) -> QCConfig:
    """Defaults < --config file < dedicated flags < --set KEY=VALUE."""
    data: dict = {}
    if args.config:
        try:
            data.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigError(f"cannot read --config {args.config}: {e}") from e
    if args.max_bad_channel_ratio is not None and args.max_bad_channels is not None:
        raise ConfigError("use either --max-bad-channel-ratio or --max-bad-channels, not both")
    flag_map = {
        "max_bad_channel_ratio": args.max_bad_channel_ratio, "max_bad_channels": args.max_bad_channels,
        "critical_channels": args.critical_channels, "line_freq": args.line_freq, "min_sfreq_hz": args.min_sfreq,
        "max_sfreq_hz": args.max_sfreq, "duration_fail_below_s": args.min_duration,
        "min_eeg_channels": args.min_eeg_channels, "closed_tasks": args.closed_tasks, "open_tasks": args.open_tasks,
    }
    data.update({k: v for k, v in flag_map.items() if v is not None})
    if args.max_bad_channel_ratio is not None:
        data["max_bad_channels"] = None                       # an explicit ratio wins over a count from the file
    for text in args.overrides:
        k, v = parse_override(text)
        data[k] = v
    base = QCConfig().to_dict()
    base.update(data)
    if "line_freq" in base:
        base["line_freq"] = str(base["line_freq"]).replace(".0", "")
    return QCConfig.from_dict(base)


def _final_report(result: BatchResult, opts: RunOptions, log: Logger) -> None:
    v = result.verdicts
    log.info("")
    log.info("=" * 78)
    log.info(f"Finished in {human_seconds(result.elapsed_s)}: {result.completed} subject(s) screened, "
             f"{result.skipped} skipped (already done), {result.partial} partial, {result.failed} failed")
    if v:
        log.info(f"Recordings this run: {v.get('PASS', 0)} PASS / {v.get('FLAG', 0)} FLAG / {v.get('FAIL', 0)} FAIL"
                 f" / {v.get('MISSING', 0)} MISSING")
    if result.bytes_downloaded:
        log.info(f"Downloaded {human_bytes(result.bytes_downloaded)}, deleted {human_bytes(result.bytes_deleted)}"
                 + (f" ({result.cleanup_errors} cleanup problem(s))" if result.cleanup_errors else ""))
    log.info(f"Results: {opts.output_dir}/qc_summary.xlsx (cumulative over all subjects in that folder)")
    for m in result.messages:
        log.warn(m)
    if result.unfinished:
        log.warn("not finished: " + ", ".join(result.unfinished[:20]) + (" ..." if len(result.unfinished) > 20 else "")
                 + ". Run the same command again to retry only these.")


def _safe_console() -> None:
    """Never crash on a console that cannot print an en dash (legacy Windows code pages): replace instead."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")            # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    log = Logger(quiet=args.quiet)

    if bool(args.dataset) == bool(args.bids_dir):
        parser.error("exactly one of --dataset DSID or --bids-dir DIR is required "
                     "(this tool never asks questions; pass everything on the command line)")
    if args.stream and (args.cleanup is False or args.keep_data):
        parser.error("--stream (Download–Screen–Delete) cannot be combined with --no-cleanup / --keep-data")
    if args.force and args.overwrite_all:
        parser.error("use either --force or --overwrite-all, not both")

    try:
        cfg = build_config(args)
    except ConfigError as e:
        log.error(str(e))
        return 2

    # ------------------------------------------------------------------ pre-flight: discovery (no EEG download)
    try:
        if args.dataset:
            log.info(f"Pre-flight: inspecting OpenNeuro dataset {args.dataset} (no EEG data is downloaded) ...")
            info = discover_openneuro(args.dataset, snapshot=args.snapshot, http=Http(), endpoints=Endpoints.from_env())
        else:
            log.info(f"Pre-flight: scanning {args.bids_dir} ...")
            info = discover_local(args.bids_dir)
    except (DatasetNotFoundError, NoEEGDataError) as e:
        log.error(str(e))
        return 2
    except DiscoveryError as e:
        log.error(str(e))
        return 1
    log.info("")
    for line in info.summary_lines():
        log.info("  " + line)

    try:
        selection = select_recordings(info, args.subjects, args.sessions, args.tasks, args.limit)
    except SelectionError as e:
        log.error(str(e))
        return 2
    n_rec = sum(len(v) for v in selection.values())
    log.info(f"  Selection    : {len(selection)} subject(s), {n_rec} recording(s)")
    if args.inspect:
        return 0

    # ------------------------------------------------------------------ options
    local = info.source == "local"
    cleanup = args.cleanup if args.cleanup is not None else (not local)
    if args.keep_data:
        cleanup = False
    if local and (args.cleanup or args.stream):
        log.warn("a local BIDS directory is read in place; its files are never deleted")
    cleanup = cleanup and not local
    work_dir = Path(args.work_dir) if args.work_dir else Path("openneuro_eeg_triage_work")
    repo_dir = Path(args.repo_dir) if args.repo_dir else work_dir / f"{info.dataset_id}_repo"
    opts = RunOptions(output_dir=Path(args.output_dir), work_dir=work_dir, repo_dir=repo_dir, cleanup=cleanup,
                      plots=args.plots, resume=args.resume, force=args.force, overwrite_all=args.overwrite_all,
                      hash_inputs=not args.no_hash, stop_after_failures=args.stop_after_failures,
                      summary_every=args.summary_every, debug=args.debug, download_method=args.download_method,
                      snapshot=args.snapshot, command=" ".join(["openneuro-eeg-triage", *(argv if argv is not None else sys.argv[1:])]))
    try:
        backend = make_backend(args.download_method, info, work_dir, repo_dir, snapshot=args.snapshot)
    except BackendError as e:
        log.error(str(e))
        return 2
    mode = ("Download–Screen–Delete (single-recording peak disk footprint)" if (cleanup and backend.deletes_data)
            else "files are kept on disk")
    log.info(f"  Mode         : {mode}; backend {backend.name}; thresholds {cfg.fingerprint()}")
    log.info("")

    try:
        result = run_batch(info, selection, backend, cfg, opts, log)
    except OutputDirError as e:
        log.error(str(e))
        return 2
    except Exception as e:                                       # noqa: BLE001
        if args.debug:
            raise
        log.error(f"{type(e).__name__}: {e}")
        return 1
    _final_report(result, opts, log)
    return result.exit_code


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
