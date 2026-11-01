"""openneuro-eeg-triage: automated EEG quality triage for OpenNeuro / BIDS datasets.

Download–Screen–Delete (single-recording peak disk footprint) with a 5-layer QC engine for standard
10-20 / 10-10 / 10-05 montages. See the README for usage.
"""
__version__ = "0.1.0"

from .config import QC_LOGIC_VERSION, QCConfig, default_config  # noqa: E402

__all__ = ["QCConfig", "QC_LOGIC_VERSION", "default_config", "__version__"]
