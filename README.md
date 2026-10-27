# openneuro-eeg-triage

> Automated EEG quality screening for OpenNeuro datasets: stream, screen, clean up.

**Status:** Planning. No code has been released yet.

## What is it?

`openneuro-eeg-triage` is a planned Python package that screens the quality of EEG recordings in [OpenNeuro](https://openneuro.org) BIDS datasets **without downloading the whole dataset**.

For each recording, the tool will:

1. **Stream**: fetch only that recording's files (plus sidecars).
2. **Screen**: run a 5-layer QC pipeline in memory.
3. **Clean up**: delete the raw files immediately, so disk usage stays small.

This makes it suitable for large datasets and for limited-storage environments such as Google Colab.

## Planned features

- Works with any OpenNeuro dataset ID (e.g. `ds005385`) or a local BIDS folder
- Pre-flight check: verifies that the dataset exists and contains EEG data
- Standard 10-20 / 10-10 / 10-05 montages only. Non-standard montages are rejected with a clear message.
- 5-layer QC: integrity, channel quality, epoch quality, spectral analysis, and artifacts (ECG/EOG/muscle)
- Resumable runs (`--resume`) after interruption
- Outputs: CSV, JSON, audit trail, diagnostic plots, and a color-coded Excel dashboard
- Interactive terminal wizard and a headless CLI

## Why?

Most EEG tools assume the data is already on disk. This project targets the step before analysis: quickly deciding which recordings in a large public dataset are usable.

## Background

The QC logic was first developed and validated on the Dortmund Vital Study (`ds005385`). This package generalizes it to other datasets.

## Roadmap

- [ ] Modular QC engine and tests
- [ ] OpenNeuro discovery and streaming
- [ ] CLI and interactive mode
- [ ] Documentation and a Colab example
- [ ] JOSS paper

## License

MIT
