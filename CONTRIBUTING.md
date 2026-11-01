# Contributing to openneuro-eeg-triage

Thank you for considering a contribution. Bug reports, datasets that break the tool, threshold feedback and pull requests
are all welcome.

## Reporting problems

Open a GitHub issue and include

* the exact command line (the tool never asks questions, so the command is the full recipe),
* the output of `openneuro-eeg-triage --version`,
* the dataset id or a description of the layout (file format, number of channels, naming of channels),
* for a wrong QC verdict: the `status.json` of the subject and, if you can share it, `qc_results.csv`.

Never attach raw EEG of people who have not consented to sharing it. A synthetic recording that reproduces the problem is ideal.

## Development setup

```bash
git clone https://github.com/alitkbbl/openneuro-eeg-triage.git
cd openneuro-eeg-triage
pip install -e ".[dev]"
pytest              # about 2 minutes, no network, synthetic data only
ruff check .
```

LibreOffice (`soffice`) is optional; when present, one test recalculates the workbook's Overview formulas.

## Ground rules

* **The tool is non-interactive.** Never call `input()` or add prompts; every choice must be a command-line flag. A test checks this.
* **Thresholds belong in `QCConfig`** (`config.py`), never inline in the engine. A new threshold needs a comment, a default, a
  validation rule if it can be invalid, and (if it can be set from the command line) a flag in `cli.py`.
* **A change to what a verdict means** (a threshold default, a rule, a reason code) must bump `QC_LOGIC_VERSION` so that old
  results are not silently reused, and must be mentioned in the pull request.
* **Raw data are never modified** and **Download–Screen–Delete must stay leak-free**: everything fetched for a recording is
  registered in the `FetchSession` the moment it arrives and deleted in the `finally:` block after the metrics are persisted.
* Every behaviour change needs a test on **synthetic data** (see `tests/conftest.py` for generators of clean and defective
  19/32/64/128-channel recordings, writers for all five file formats and a fake OpenNeuro server).
* Keep the scope: standard 10-20 / 10-10 / 10-05 montages. Support for other layouts should be discussed in an issue first.

## Pull requests

1. Fork, create a branch, make focused commits.
2. `pytest` and `ruff check .` pass.
3. Describe the motivation, and for QC changes the effect on at least one synthetic or public recording.

By contributing you agree that your contribution is released under the MIT License.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
