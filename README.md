# openneuro-eeg-triage

**Automated EEG quality triage for OpenNeuro and BIDS datasets: download, screen, delete.**

Give it any OpenNeuro dataset id (for example `ds005385`) or a local BIDS folder. It inspects the dataset
*without downloading any EEG data*, then screens the recordings **one at a time** with a 5-layer quality engine and
deletes each recording's raw files before fetching the next. Local and Colab disks never fill up, and the results
accumulate into one globally ranked Excel workbook.

```bash
pip install openneuro-eeg-triage          # or: pip install -e ".[dev]" from a clone

openneuro-eeg-triage --dataset ds005385 --inspect                    # what is in it? (no EEG download)
openneuro-eeg-triage --dataset ds005385 --subjects 1 to 20 --stream  # screen subjects 1-20
openneuro-eeg-triage --dataset ds005385 --subjects 21 to 30          # later: extends and re-ranks the same results
```

It is **100 % non-interactive**: every choice is a command-line flag and the tool never pauses to ask a question,
so it runs unattended in scripts, cron jobs, Slurm and Colab.

---

## Contents

1. [Why](#1-why) · 2. [Scope: standard montages only](#2-scope-standard-10-20--10-10--10-05-montages-only) ·
3. [Download–Screen–Delete](#3-downloadscreendelete-single-recording-peak-disk-footprint) · 4. [The 5 layers](#4-the-5-qc-layers) ·
5. [Adaptive rules](#5-rules-that-adapt-to-the-recording) · 6. [Command line](#6-command-line) · 7. [Outputs](#7-outputs) ·
8. [Cumulative runs](#8-cumulative-runs-and-resuming) · 9. [Colab](#9-google-colab) · 10. [Dortmund benchmark](#10-benchmark-the-dortmund-vital-study-ds005385) ·
11. [Testing](#11-testing) · 12. [Limitations](#12-limitations) · 13. [Citation, license](#13-citation-and-license)

---

## 1. Why

Large public EEG datasets contain recordings that are fine, recordings that are easily repaired, and a few that are
broken. Before any real analysis you need a quick, reproducible answer to **"which recordings can I use?"** Looking at
hundreds of participants by eye is slow and subjective, and strict automatic rules throw away good data.

`openneuro-eeg-triage` is a **triage**, not strict QC. It keeps everything healthy or repairable and rejects only what is
catastrophic, with a short **reason code for every decision**. A recording is *not* rejected only because

* the person is an **alpha non-responder** (weak alpha with eyes closed is common in healthy adults, so it is a `FLAG`);
* the signal drifts (statistics use a baseline-corrected copy, the raw signal is never modified);
* there is **mains noise** at 50 or 60 Hz (a notch filter fixes it, so it is a `FLAG`);
* a handful of channels are bad (they can be interpolated, so the recording is `USABLE_NEEDS_INTERPOLATION`).

Most EEG tools assume the data are already on disk. This one targets the step *before* analysis.

## 2. Scope: standard 10-20 / 10-10 / 10-05 montages only

Only caps labelled with the **international 10-20, 10-10 or 10-05 systems** are supported, at any density (19, 21, 32,
64, 128 channels ...). There is no hard-coded channel count.

* **Name normalisation:** `EEG Fz-REF` → `Fz`, `fp1` → `Fp1`, `OZ` → `Oz`, and legacy 10-20 names `T3/T4/T5/T6` → `T7/T8/P7/P8`.
* **Standard-montage gatekeeper (Layer 1):** at least 70 % of the EEG channels must carry a standard name *and* at least
  6 of the 10 core anchors (`Fz Cz Pz Oz F3 F4 C3 C4 O1 O2`) must be verifiable.
* **Anything else is rejected cleanly, never guessed:** EGI geodesic nets (`E1..E128`), unmapped BioSemi (`A1..A32`) and other
  proprietary labels give `UNSUPPORTED_MONTAGE` in Layer 1. When the dataset ships a BIDS `channels.tsv`, this is decided
  **before the large binary file is downloaded**.
* **Classic 19-channel caps have no `Oz`.** The posterior alpha region of interest is `O1, O2, Oz` when `Oz` exists and falls
  back to `O1, O2` otherwise. `Oz` is never required; `O1` and `O2` are.

Supported file formats: **EDF, BDF, BrainVision (`.vhdr` + `.vmrk` + `.eeg`/`.dat`), EEGLAB (`.set`, with optional `.fdt`) and
FIF**, in single-session (`sub-X/eeg/`) or multi-session (`sub-X/ses-Y/eeg/`) layouts with optional `acq-*` and `run-*` entities.

## 3. Download–Screen–Delete (single-recording peak disk footprint)

For every selected recording the tool

1. fetches the small sidecars (`*_eeg.json`, `*_channels.tsv`, `*_events.tsv`) first;
2. fetches the main file **and all companion files** (`.vhdr`+`.vmrk`+`.eeg`, `.set`+`.fdt`; the names in the BrainVision header's
   `DataFile=` / `MarkerFile=` lines are honoured);
3. screens it in memory and **persists the metrics**;
4. deletes everything it fetched, in a `finally:` block, so also after an error.

At any moment only **one recording** of the dataset is on disk. The raw data are never modified. With `--bids-dir` the files are
read in place and **never deleted**.

Backends (`--download-method`): `https` pulls from OpenNeuro's public S3 bucket and needs only Python; `datalad` uses
`datalad get` / `datalad drop` and pins an exact snapshot (`--snapshot`); `auto` (default) uses DataLad if it and git-annex
are installed, otherwise HTTPS. Note that HTTPS always serves the *newest* snapshot.

Discovery (`--inspect`) uses the dataset's git tree on the `OpenNeuroDatasets` GitHub mirror (an exact snapshot tag), with the
public S3 bucket listing as a fallback. Set `GITHUB_TOKEN` if GitHub rate-limits you.

## 4. The 5 QC layers

Every finding has a verdict and a short reason code. **PASS** nothing to note · **FLAG** small problem, recording kept ·
**FAIL** clear problem, not usable · **MISSING** not available (never stops the run).

| # | Layer | Checks | FAIL | FLAG |
|---|---|---|---|---|
| 1 | **Header & metadata** | EEG channel count, sampling rate, duration, unique names, **standard montage**, `O1`/`O2` present, JSON sidecar | non-standard montage, fewer than `min_eeg_channels` (16), sfreq below `min_sfreq_hz` (128), shorter than 60 s, duplicate/empty names, `O1`/`O2` missing | duration outside 2 min – 1 h, sampling rate above `max_sfreq_hz`, sidecar differs from the file |
| 2 | **Channel integrity** | flat, clipped, noisy and **spatially uncorrelated** channels; frozen segments | see below | one critical channel bad; frozen segment over 5 s |
| 3 | **Amplitude** | typical amplitude (median 1-s STD of the baseline-corrected signal) | median outside 1–100 µV | more than 10 % of channels above 100 µV |
| 4 | **Spectral** | alpha peak, Eyes Closed / Eyes Open alpha blocking, mains line noise (50 or 60 Hz) | dead spectrum on the posterior ROI; line noise **together with** clipping of half the channels | weak alpha, no alpha blocking, line noise |
| 5 | **Consistency** | same channel names, order and sampling rate in all recordings of a subject | differs | |

**Bad-channel decision logic** (Layer 2). Critical channels default to `Cz, Fz, Pz, O1, O2` (`--critical-channels`).

| Situation | Verdict | Code |
|---|---|---|
| 2 or more critical channels bad | FAIL | `CRITICAL_CHANNELS_BAD` |
| more bad channels than the allowed maximum (below) | FAIL | `TOO_MANY_BAD_CHANNELS` |
| 1 critical channel bad | FLAG (still usable after interpolation) | `CRITICAL_CHANNEL_BAD` |
| 1 to *max* bad channels, at most one critical | PASS, label *needs interpolation* | `USABLE_NEEDS_INTERPOLATION` |

Each recording gets a label (`USABLE`, `USABLE_WITH_FLAGS`, `USABLE_NEEDS_INTERPOLATION`, `REJECT`); a subject gets the most
demanding label of its recordings, or `PARTIALLY_USABLE` when some are usable and others rejected. The full list of reason codes
is in the `Legend` sheet of the workbook.

**Eyes Closed / Eyes Open.** The condition is inferred from the `task-*` and `acq-*` labels (`closed`, `ec`, `EyesClosed` →
closed; `open`, `eo` → open) and recordings are paired on all *other* entities (session, acq/task, run). A bare `rest` is
`unknown`: the eye state cannot be known, so no alpha flag is raised for it. Override with `--closed-tasks` / `--open-tasks`.
Alpha prominence is judged on Eyes-Closed recordings only; a missing partner is only flagged when the subject has both
conditions.

## 5. Rules that adapt to the recording

| Quantity | Rule |
|---|---|
| Max bad channels | `max(1, floor(max_bad_channel_ratio × n_eeg))`, ratio default **0.10** → 64 ch: 6, 65 ch: 6, 32 ch: 3, 19 ch: 1. Override with `--max-bad-channel-ratio R` **or** an exact `--max-bad-channels N`. |
| Spatial-correlation thresholds (0.20; 0.18 for O1/O2/Oz) | multiplied by `sqrt(n_eeg / 64)`, never below 0.6 and never above 1, so sparse caps are not penalised by 64-channel assumptions (19 ch → ×0.60, 32 ch → ×0.71). The "reference not trusted" safeguard (more than half of the channels below threshold → nobody is marked bad, `LOW_GLOBAL_SPATIAL_CORRELATION`) is a fraction already. |
| High-frequency bound | `min(100 Hz, 0.9 × Nyquist)` when `sfreq ≤ 250 Hz` (correlation band, dead-spectrum band, harmonics); a line whose flank would exceed 98 % of Nyquist uses its lower flank only, or is reported as not assessable. |
| Mains frequency | `--line-freq 50\|60` > `PowerLineFrequency` in `*_eeg.json` > spectral auto-detection (a candidate must stand out by ≥ 6 dB, otherwise 50 Hz is reported with source `default_no_line_detected`). |
| Frozen-segment length | `frozen_min_s` = 0.1 s (100 samples at 1 kHz) at any sampling rate. |

Every threshold lives in `QCConfig` ([`config.py`](src/openneuro_eeg_triage/config.py)) and can be changed with a dedicated flag,
`--config FILE.json`, or `--set KEY=VALUE` (repeatable). A fingerprint of all thresholds is stored with every result.

## 6. Command line

```text
openneuro-eeg-triage (--dataset DSID | --bids-dir DIR) [options]
```

| Group | Flags |
|---|---|
| Source | `--dataset DSID`, `--bids-dir DIR`, `--snapshot TAG` |
| Selection | `--subjects 28` · `1 2 3` · `1 to 20` · `1-20` · `5,9` · `sub-PD01` (mixable), `--sessions`, `--tasks`, `--limit N`, `--inspect` |
| Output / run | `--output-dir/-o`, `--work-dir`, `--repo-dir`, `--download-method auto\|https\|datalad`, `--stream`, `--cleanup/--no-cleanup`, `--keep-data`, `--plots/--no-plots`, `--resume/--no-resume`, `--force`, `--overwrite-all`, `--no-hash`, `--stop-after-failures N`, `--summary-every N`, `--quiet`, `--debug` |
| Thresholds | `--max-bad-channel-ratio R`, `--max-bad-channels N`, `--critical-channels CH ...`, `--line-freq auto\|50\|60`, `--min-sfreq HZ`, `--max-sfreq HZ`, `--min-duration S`, `--min-eeg-channels N`, `--closed-tasks`, `--open-tasks`, `--config FILE`, `--set KEY=VALUE` |

Run `openneuro-eeg-triage --help` for the full text. Exit codes: **0** finished · **1** technical problem (network, unfinished
subjects, Ctrl+C) · **2** invalid input (bad flags, unknown dataset, no EEG data, empty selection).

**Pre-flight.** Before anything is downloaded the tool tells you whether the dataset exists, whether it contains EEG at all
(an MRI-only dataset exits with code 2 and says which modalities it found), and prints an executive summary: title, subjects,
sessions, tasks, file formats and the number of recordings.

## 7. Outputs

```text
qc_output/
├── qc_summary.xlsx          6-sheet workbook over ALL subjects in this folder (rebuilt on every run)
├── qc_results.csv           one row per recording, every metric of every layer, SHA-256 of the inputs
├── qc_subjects.csv          one row per subject
├── qc_summary.json          counts, dataset, version, the complete thresholds
├── audit_trail.jsonl        append-only log: start, downloaded, screened, cleaned, errors
├── .openneuro_eeg_triage.json   marker: which dataset this folder belongs to
└── sub-XXX/
    ├── status.json          subject status + final per-recording results
    ├── recordings/          cached per-recording results (this is what --resume reads)
    └── figures/             posterior spectra, channel-quality topomap, amplitude heatmap
```

**`qc_summary.xlsx`** has six sheets: **Overview** (live formulas on the other sheets and charts), **Subjects** (one row per
subject, with a global rank), **Recordings** (one row per recording), **Recording_Ranking** (all recordings ranked best to worst
by a 0–100 quality score; rejected recordings always last), **Issues** (every finding) and **Legend** (statuses, verdicts, labels,
reason codes, score definition, the thresholds used). The quality score uses seven signal criteria and only orders the
recordings; it never changes a verdict. Electrode positions in the topomap come from MNE's standard template, not from
individual digitisation.

## 8. Cumulative runs and resuming

The workbook and CSV/JSON summaries are rebuilt from **every** `sub-*` folder in the output directory. Screen 20 subjects, then
10 more, and the second run produces one complete workbook over all 30 subjects, with `Overview` recomputed and `Subjects` and
`Recording_Ranking` re-ranked across all of them.

* Re-running the same command skips finished subjects and retries only unfinished ones (`--resume`, on by default). A subject is
  *finished* when it was screened with the same QC logic and thresholds (a fingerprint is stored). Ctrl+C or a crash never
  destroys finished work; per-recording results are cached.
* A download that fails marks the recording `MISSING`, the subject `PARTIAL` / `NO_DATA`, and is retried next time. After 3
  subjects in a row with failed downloads the batch stops by itself (`--stop-after-failures`).
* `--force` re-screens the *selected* subjects and keeps and merges the others; `--overwrite-all` deletes all previous results in
  the output directory first.
* Results screened with different thresholds are kept but marked `config_stale` in the `Subjects` sheet, with a warning.
* One output directory belongs to one dataset; mixing datasets is refused.

## 9. Google Colab

```python
!pip install -q openneuro-eeg-triage
!openneuro-eeg-triage --dataset ds005385 --subjects 1 to 50 --stream --output-dir /content/drive/MyDrive/qc_output
```

Mount Drive first (`from google.colab import drive; drive.mount('/content/drive')`) so results survive a disconnect, and run
the same command again in a new session to continue.

## 10. Benchmark: the Dortmund Vital Study (ds005385)

The QC logic was first developed as a single-dataset prototype (`qc_eeg_subject.py`, QC version `screening-2.2`) on the Dortmund
Vital Study resting-state EEG ([ds005385](https://openneuro.org/datasets/ds005385), 64 channels, 1 kHz, Eyes Closed / Eyes
Open before and after a cognitive task, 465 GB). The prototype applied project-specific inclusion rules (age and trigger
criteria, not part of this package) that select 427 of the 608 participants listed in `participants.tsv`, which implies
2,276 expected recordings (142 two-session participants × 8 + 285 one-session participants × 4).

**Pre-flight check on the real dataset.** Running discovery on the git tree of snapshot 1.0.3 shows EEG files for only
**374 of the 608 participants (1,998 recordings, all EDF; 266 of the 427 prototype-eligible participants)**. The number of
recordings actually present therefore has to be taken from the dataset, not from the participant table. This is exactly the
kind of fact `--inspect` reports before anything is downloaded. The real `channels.tsv` of the dataset (64 labels) is accepted by
the standard-montage gatekeeper and is used as a test fixture (`tests/data/ds005385_channels.tsv`).

This package generalises the prototype: the channel-count and sampling-rate locks are gone, thresholds adapt to the cap, and the
Dortmund-specific participant rules were removed. At 64 channels and 1 kHz the defaults (maximum 6 bad channels, 5 critical
channels, all thresholds) give **numerically identical** results to the prototype on synthetic data: every column of the channel
table, the integrity and amplitude verdicts and all alpha and line-noise metrics were compared and are bit-identical
(`tests/test_prototype_regression.py`). **A formal re-run of the whole dataset with this package and a recording-by-recording
comparison with the prototype's results on real data is still to be done**; small differences are expected there (for example
`Oz` is no longer required, the duration FLAG range is wider and the mains frequency is detected rather than fixed at 50 Hz).

```bash
openneuro-eeg-triage --dataset ds005385 --inspect                          # what is really in the snapshot
openneuro-eeg-triage --dataset ds005385 --stream -o dortmund_qc            # screen everything that is there
```

## 11. Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite uses **synthetic recordings only** (no network, no real data): clean and defective 19-, 32-, 64- and 128-channel
caps, legacy channel names, 50 and 60 Hz line noise, flat/clipped/noisy/uncorrelated channels, frozen segments, scale errors,
non-standard montages, all five file formats (including a hand-written BDF), and end-to-end runs of the CLI against a local
fake OpenNeuro server that exercises the real HTTP code, including multi-batch cumulative re-ranking, resume, force, overwrite,
failed downloads and Ctrl+C. The Overview formulas are recalculated with LibreOffice when it is installed. The thresholds are
practical heuristics and have **not** been validated against manual ratings.

## 12. Limitations

* This is a **first-pass screening**, not a replacement for expert review or a full preprocessing pipeline. The default
  thresholds are heuristics from one dataset; check them on a few recordings of your own and adjust.
* The **spatial-correlation criterion is the most data-dependent one**. Referencing, the reference electrode position and
  frontal eye activity can lower the correlation of individual channels. Independent 1/f noise correlates with the median
  reference at roughly |r| = 0.1–0.17, close to the default thresholds, so the criterion cannot separate every disconnected
  channel from a poor one. The channel-count scaling in section 5 is a heuristic.
* Only signal quality is judged. Trigger codes and the cognitive task are not interpreted. ECG/EOG channels are recognised and
  excluded from the EEG analysis but not analysed.
* HTTPS downloads serve OpenNeuro's newest snapshot; use DataLad for an exactly pinned version.
* Non-standard montages are rejected, not interpolated onto a template.

## 13. Citation and license

If you use this software, please cite it (see [`CITATION.cff`](CITATION.cff) and the JOSS paper in [`paper/`](paper/paper.md))
and the dataset you screened. Released under the [MIT License](LICENSE). Contributions are welcome, see
[CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

Author: Ali Taghizadeh · <alitaghi8305@gmail.com>
