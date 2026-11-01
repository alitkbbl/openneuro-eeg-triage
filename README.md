# OpenNeuro EEG Triage

<p align="left">
  <img src="https://img.shields.io/badge/EEG-Quality%20Triage-6f42c1?style=flat" alt="EEG Quality Triage" />
  <img src="https://img.shields.io/badge/OpenNeuro-compatible-2563eb?style=flat" alt="OpenNeuro compatible" />
  <img src="https://img.shields.io/badge/BIDS-supported-0f9d58?style=flat" alt="BIDS supported" />
  <img src="https://img.shields.io/badge/Workflow-Streaming-f59e0b?style=flat" alt="Streaming workflow" />
  <img src="https://img.shields.io/badge/License-MIT-64748b?style=flat" alt="MIT License" />
</p>

**Automated quality triage for OpenNeuro and local BIDS EEG datasets.**

Screen large EEG datasets without filling your disk: inspect metadata first, process recordings **one at a time**, and automatically clean up downloaded files. All results are collected in a cumulative Excel report with global quality rankings.

> **Inspect → Download → Screen → Clean up → Rank**  
> Designed for reproducible, unattended runs on local machines, servers, and Google Colab.

## 🚀 Quick start

``` bash
pip install openneuro-eeg-triage

# Inspect dataset contents without downloading EEG data
openneuro-eeg-triage --dataset ds005385 --inspect

# Screen selected subjects
openneuro-eeg-triage --dataset ds005385 --subjects 1 to 20 --stream

# Add more subjects; existing results are retained and re-ranked
openneuro-eeg-triage --dataset ds005385 --subjects 21 to 30
```

The CLI is fully non-interactive and suitable for scripts, scheduled
jobs, Slurm, and Google Colab.

## 🧠 What it does

The tool is designed for **first-pass triage**, not strict QC. It aims
to retain recordings that are healthy or repairable and reject only
clearly unusable data. Every decision includes a reason code.

A recording is not rejected solely because it has:

-   Weak posterior alpha or no clear alpha blocking (`FLAG`).
-   Signal drift (statistics use a baseline-corrected copy; raw data
    remain unchanged).
-   50/60 Hz mains noise (`FLAG`; can be handled during preprocessing).
-   A small number of bad channels (`USABLE_NEEDS_INTERPOLATION`).

## 📁 Supported data

-   **Montages:** International 10-20, 10-10, and 10-05 systems, at any
    channel density.
-   **Formats:** EDF, BDF, BrainVision, EEGLAB (`.set` with optional
    `.fdt`), and FIF.
-   **Layouts:** Single- and multi-session BIDS EEG directories.
-   **Sources:** OpenNeuro dataset IDs or local BIDS folders.

Channel names are normalised (for example, `EEG Fz-REF` to `Fz`, and
`T3/T4/T5/T6` to `T7/T8/P7/P8`). At least 70% of EEG channels must have
standard names, and at least 6 of 10 core electrodes must be
identifiable. Unsupported or ambiguous montages are rejected as
`UNSUPPORTED_MONTAGE`; the tool does not guess mappings. `O1` and `O2`
are required; `Oz` is optional.

## ♻️ Download–screen–delete workflow

For each recording, the tool:

1.  Downloads BIDS sidecars first (`*_eeg.json`, `*_channels.tsv`,
    `*_events.tsv`).
2.  Downloads the EEG file and its required companion files.
3.  Screens the recording and saves its metrics.
4.  Deletes downloaded files, including after an error.

Only one downloaded recording is kept on disk at a time. Local BIDS
files are read in place and are never deleted.

Download methods:

-   `auto` (default): DataLad when available; otherwise HTTPS.
-   `https`: downloads from OpenNeuro's public S3 bucket; uses the
    latest snapshot.
-   `datalad`: supports an exact snapshot via `--snapshot`.

Dataset discovery uses the OpenNeuro GitHub mirror, with S3 listing as a
fallback. Set `GITHUB_TOKEN` if GitHub rate limits are encountered.

## 🔬 QC pipeline

Each recording is assessed across five layers:

| Layer | Main checks |
|---|---|
| **1. Header & metadata** | Channel count, sampling rate, duration, channel names, montage, `O1`/`O2`, and sidecar consistency |
| **2. Channel integrity** | Flat, clipped, noisy, spatially uncorrelated channels, and frozen segments |
| **3. Amplitude** | Typical signal amplitude using the median 1-second standard deviation |
| **4. Spectral** | Posterior alpha, eyes-closed/open alpha blocking, and 50/60 Hz line noise |
| **5. Consistency** | Channel names, order, and sampling rate across a subject's recordings |

Findings are labelled `PASS`, `FLAG`, `FAIL`, or `MISSING`. Missing
information does not stop a run.

Bad-channel handling uses configurable critical channels (default:
`Cz, Fz, Pz, O1, O2`). Two or more bad critical channels, or too many
bad channels overall, cause rejection. A recording with a limited number
of bad channels can remain usable after interpolation.

Recording labels are `USABLE`, `USABLE_WITH_FLAGS`,
`USABLE_NEEDS_INTERPOLATION`, or `REJECT`. Subject-level status reflects
the most demanding recording outcome, or `PARTIALLY_USABLE` when usable
and rejected recordings coexist. See the workbook's **Legend** sheet for
all reason codes.

Eyes-closed/open conditions are inferred from `task-*` and `acq-*`
labels. A plain `rest` label is treated as unknown. Use `--closed-tasks`
and `--open-tasks` to override detection.

## ⚙️ Adaptive thresholds

Several rules scale with the recording:

-   Maximum bad channels defaults to 10% of EEG channels, with a minimum
    of one.
-   Spatial-correlation thresholds are adjusted for channel count.
-   High-frequency analysis respects the Nyquist frequency.
-   Mains frequency is taken from `--line-freq`, BIDS metadata, or
    spectral detection.
-   Frozen segments use a 0.1-second minimum.

All thresholds are configurable through `QCConfig`,
`--config FILE.json`, or repeatable `--set KEY=VALUE` options. The
threshold fingerprint is stored with each result.

## 💻 Command line

``` text
openneuro-eeg-triage (--dataset DSID | --bids-dir DIR) [options]
```

Common options:

| Purpose | Options |
|---|---|
| **Source** | `--dataset`, `--bids-dir`, `--snapshot` |
| **Select data** | `--subjects`, `--sessions`, `--tasks`, `--limit`, `--inspect` |
| **Run and output** | `--output-dir` / `-o`, `--work-dir`, `--download-method`, `--stream`, `--resume`, `--force`, `--overwrite-all` |
| **QC settings** | `--max-bad-channel-ratio`, `--max-bad-channels`, `--critical-channels`, `--line-freq`, `--min-sfreq`, `--max-sfreq`, `--min-duration`, `--min-eeg-channels`, `--config`, `--set` |

For all options, run `openneuro-eeg-triage --help`.

Before downloading, `--inspect` reports dataset metadata, available
modalities, subjects, sessions, tasks, file formats, and recording
counts. Invalid inputs exit with code `2`; technical or incomplete runs
exit with code `1`; successful runs exit with code `0`.

## 📊 Outputs

The output directory contains:

``` text
qc_output/
├── qc_summary.xlsx       # Six-sheet report and global rankings
├── qc_results.csv        # One row per recording, with QC metrics
├── qc_subjects.csv       # One row per subject
├── qc_summary.json       # Dataset, counts, version, thresholds
├── audit_trail.jsonl     # Append-only run and cleanup log
├── .openneuro_eeg_triage.json
└── sub-XXX/
    ├── status.json
    ├── recordings/       # Cached results used by --resume
    └── figures/          # Spectra, topomaps, amplitude heatmaps
```

`qc_summary.xlsx` includes **Overview**, **Subjects**, **Recordings**,
**Recording_Ranking**, **Issues**, and **Legend** sheets. Recordings are
ranked by a 0--100 quality score; the score orders results but does not
change QC verdicts.

## 🔁 Resuming and cumulative runs

All summaries are rebuilt from the subject folders in the output
directory. You can screen subjects in batches and the workbook will
include and re-rank all completed results.

-   `--resume` is enabled by default and skips completed subjects with
    matching QC settings.
-   Interrupted runs retain completed recording results.
-   Failed downloads are marked missing and can be retried.
-   `--force` re-screens selected subjects; `--overwrite-all` clears
    previous output.
-   Results from older settings are retained but marked `config_stale`.
-   Each output directory is tied to one dataset.

## ☁️ Google Colab

Mount Google Drive, then install and run:

``` python
from google.colab import drive
drive.mount('/content/drive')
```

``` bash
!pip install -q openneuro-eeg-triage
!openneuro-eeg-triage --dataset ds005385 --subjects 1 to 50 --stream \
  --output-dir /content/drive/MyDrive/qc_output
```

Run the same command in a later session to resume. Saving results to
Drive protects them from Colab session disconnections.

## 🧪 Dortmund Vital benchmark (`ds005385`)

The package generalises an earlier project-specific screening prototype
developed on the Dortmund Vital Study (64 channels, 1 kHz, 465 GB).

For OpenNeuro snapshot `1.0.3`, dataset discovery found EEG files for
**374 of 608 participants**, totalling **1,998 EDF recordings**. This
differs from counts inferred from `participants.tsv`, illustrating why
`--inspect` checks the actual dataset contents before downloading.

At 64 channels and 1 kHz, default settings produce bit-identical results
to the prototype on synthetic regression data. A full real-data re-run
and recording-by-recording comparison have **not yet been completed**.

``` bash
openneuro-eeg-triage --dataset ds005385 --inspect
openneuro-eeg-triage --dataset ds005385 --stream -o dortmund_qc
```

## 🧰 Testing

``` bash
pip install -e ".[dev]"
pytest
```

Tests use synthetic recordings and a local mock OpenNeuro server. They
cover multiple channel densities, supported file formats, common signal
defects, cumulative runs, resume behaviour, failed downloads, and
interruption handling.

The default thresholds are practical heuristics and have not been
validated against expert manual ratings.

## ⚠️ Limitations

-   This is first-pass screening, not expert review or a preprocessing
    pipeline.
-   Default thresholds were developed from limited data; inspect sample
    recordings and tune them for each dataset.
-   Spatial correlation is sensitive to referencing, electrode
    locations, and frontal eye activity.
-   The tool evaluates signal quality; it does not interpret event
    triggers or task performance.
-   ECG/EOG channels are excluded from EEG analysis but are not
    independently assessed.
-   HTTPS uses the newest OpenNeuro snapshot; use DataLad to pin a
    version.
-   Non-standard montages are rejected rather than automatically
    remapped.

## 📚 Citation, license, and contribution

If you use this software, cite the package (see
[`CITATION.cff`](CITATION.cff) and [`paper/`](paper/)) and the dataset
you screened.

Released under the [MIT License](LICENSE). Contributions are welcome;
see [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

**Author:** Ali Taghizadeh · <alitaghi8305@gmail.com>
