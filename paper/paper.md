---
title: 'openneuro-eeg-triage: EEG quality triage for OpenNeuro datasets with a single-recording peak disk footprint'
tags:
  - Python
  - EEG
  - quality control
  - BIDS
  - OpenNeuro
  - open data
authors:
  - name: Ali Taghizadeh
    orcid: 0000-0000-0000-0000   # TODO: replace with your ORCID
    affiliation: 1
affiliations:
  - name: TODO institution, country
    index: 1
date: 30 October 2026
bibliography: paper.bib
---

<!-- Draft. Items marked TODO need the author's input before submission; see the final section. -->

# Summary

Public repositories such as OpenNeuro [@markiewicz2021openneuro] now hold hundreds of electroencephalography (EEG) datasets in
the Brain Imaging Data Structure (BIDS) [@gorgolewski2016bids; @pernet2019eegbids]. Before reusing one, a researcher needs a
fast, reproducible answer to a simple question: which recordings are usable? `openneuro-eeg-triage` answers it for any
BIDS-EEG dataset, identified by an OpenNeuro accession number or a local folder, without requiring the dataset to fit on the
local disk. It inspects the dataset's contents without downloading EEG data, then screens each recording with a five-layer
quality engine and deletes the raw files before fetching the next recording. Results from any number of runs accumulate into one
globally ranked Excel workbook, CSV and JSON tables, an audit trail and diagnostic figures. The tool is entirely
command-line driven and never asks interactive questions, so it runs unattended on laptops, clusters and Google Colab.

# Statement of need

EEG preprocessing and artefact-handling tools such as MNE-Python [@gramfort2013mne], the PREP pipeline and its Python port
PyPREP [@bigdelyshamlo2015prep; @appelhoff2022pyprep] and Autoreject [@jas2017autoreject] detect and repair bad channels or
epochs *within an analysis*, and MNE-BIDS [@appelhoff2019mnebids] reads BIDS-EEG data. All of them assume that the data are
already on disk, and none is designed to produce a dataset-wide, ranked usability overview. Downloading a dataset such as the
Dortmund Vital Study (608 participants, about 465 GB) merely to discover which recordings are unusable is costly, and
unavailable on storage-limited machines. Manual inspection is slow and subjective, while strict automatic rejection rules
discard recordings that are healthy or easily repaired, for instance those of alpha non-responders, those with mains noise that a
notch filter removes, or those with a few interpolatable channels.

`openneuro-eeg-triage` fills the step *before* analysis. It is a *triage*, not strict quality control: it keeps everything
healthy or repairable and rejects only what is catastrophic, attaching a short reason code to every decision so that users can
audit and override it. <!-- TODO: before submission, verify this positioning with a literature and software search; other QC tools may exist. -->

# Software design

**Download–Screen–Delete (single-recording peak disk footprint).** For each selected recording the tool fetches the small BIDS
sidecars, then the main file and all its companion files (BrainVision `.vhdr`, `.vmrk` and `.eeg`/`.dat`; EEGLAB `.set` and
`.fdt`), screens the recording in memory, persists its metrics, and deletes everything it fetched in a `finally` block. At any
moment one recording of the dataset is on disk. Files come from OpenNeuro's public S3 bucket with a pure-Python HTTPS client, or
through DataLad [@halchenko2021datalad] where an exactly pinned snapshot is needed. Dataset discovery uses the dataset's git
tree on the OpenNeuro GitHub mirror, with the S3 listing as a fallback, so the subjects, sessions, tasks, runs and file formats
are known before any EEG binary is requested.

**Standard-montage architecture.** The package supports caps labelled with the international 10-20, 10-10 and 10-05 systems at
any density. Channel labels are normalised (hardware prefixes and suffixes removed, case fixed, legacy names `T3/T4/T5/T6`
mapped to `T7/T8/P7/P8`), and a gatekeeper requires a standard-name fraction and a set of core anchor electrodes. Recordings with
proprietary labels (for example EGI nets or unmapped BioSemi channels) are rejected with a diagnostic code rather than guessed
at, and when a `channels.tsv` is available this happens before the data file is downloaded. Thresholds adapt to the recording: the
number of channels that may be interpolated is `max(1, floor(r n))` for `n` EEG channels and a configurable ratio `r`
(default 0.10), correlation thresholds are scaled for sparse caps, analysis bands respect the Nyquist frequency, and the mains
frequency (50 or 60 Hz) is read from the sidecar or detected from the spectrum.

**Five quality layers.** (1) header and metadata, including the montage check; (2) channel integrity: flat, clipped, noisy and
spatially uncorrelated channels, assessed against a median reference, and frozen segments, with a decision rule that separates
repairable from unusable recordings through a set of critical channels; (3) amplitude on a baseline-corrected copy of the signal, so
slow drift cannot masquerade as noise; (4) spectral: alpha peak, eyes-closed/eyes-open alpha blocking when both conditions are
available, and line noise, where weak alpha is only a flag; (5) consistency of channel layout and sampling rate across a subject's
recordings. Every finding is a `PASS`, `FLAG` or `FAIL` with a reason code, and each recording receives one of four labels from
`USABLE` to `REJECT`.

**Cumulative, resumable reporting.** Each recording's result is cached and each subject has a status file. The workbook,
CSV/JSON summaries and a 0–100 quality ranking are rebuilt from all subjects in the output directory on every run, so screening
20 subjects and later 10 more yields one workbook over 30 subjects, re-ranked together. Re-running a command skips finished work, a
fingerprint of the thresholds prevents silently mixing results produced with different settings, and SHA-256 hashes of the inputs
and an append-only audit trail support reproducibility.

# Validation and current status

The screening logic originates from a single-dataset prototype that was developed on the Dortmund Vital Study (OpenNeuro
ds005385) [@getzmann2024dortmund], a 64-channel, 1 kHz resting-state dataset. Pre-flight discovery of snapshot 1.0.3 shows
EEG files for 374 of the 608 participants in `participants.tsv` (1,998 recordings; git tree of the OpenNeuro GitHub mirror,
retrieved 30 October 2026), so the recording count must be read from the dataset rather than inferred from the participant
table. The package removes the prototype's project-specific participant rules and its hard-coded 64-channel, 1 kHz
assumptions. At 64 channels and 1 kHz, every column of the per-channel table, the integrity and amplitude results and all
alpha and line-noise metrics computed by the package are identical to those of the prototype on identical synthetic data
(a regression test), and the dataset's real channel labels are accepted by the standard-montage check.
<!-- TODO: (1) confirm the 374-participant count against the OpenNeuro web interface and your prototype's output (how many
recordings did the prototype actually analyse, and how many were MISSING?). (2) Re-run ds005385 with this package and
report the agreement of recording-level labels with the prototype on real data, the distribution of labels and reason codes,
and run time and peak disk use. Do not submit with synthetic-data equivalence alone. -->

The automated test suite (`pytest`, run on Python 3.10–3.12 in continuous integration) uses synthetic recordings only: clean and
defective 19-, 32-, 64- and 128-channel caps; flat, clipped, noisy and uncorrelated channels; frozen segments; scale errors; 50
and 60 Hz noise; non-standard montages; all five supported file formats; and end-to-end command-line runs against a local server
that imitates the OpenNeuro endpoints, including cumulative multi-batch ranking, resuming, failed downloads and interruption.
This verifies that the software behaves as specified; it does not establish that the thresholds agree with expert judgement.
<!-- TODO: validate against manual ratings of a sample of recordings and/or compare bad-channel detection with PyPREP; add a
threshold-sensitivity analysis; test on at least three further OpenNeuro datasets with different caps, mains frequencies and file
formats, and report the results here. -->

# Limitations

The thresholds are practical heuristics. The spatial-correlation criterion is the most data-dependent one, and its scaling for
sparse caps is a heuristic. Only signal quality is assessed; triggers and task performance are not interpreted. Non-standard
montages are rejected rather than mapped to a template.

# AI usage disclosure

<!-- TODO: JOSS asks authors to disclose generative-AI use; check the current policy and complete this section accurately. -->
Generative AI (Anthropic's Claude) was used to assist with refactoring the prototype into a package and with drafting tests and
documentation. [TODO: author to confirm scope, and state how the output was reviewed and validated.]

# Acknowledgements

TODO: funding and acknowledgements.

# References
