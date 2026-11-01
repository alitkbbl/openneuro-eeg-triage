"""Tiny dependency-free HTTP layer (urllib only) with retries. Endpoints can be overridden for mirrors/tests."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

USER_AGENT = "openneuro-eeg-triage (+https://github.com/alitkbbl/openneuro-eeg-triage)"


@dataclass(frozen=True)
class Endpoints:
    """Where OpenNeuro metadata and files come from. All three can be overridden with environment variables
    (``OPENNEURO_EEG_TRIAGE_GITHUB_API``, ``..._GITHUB_RAW``, ``..._S3_BASE``), e.g. for a mirror or a test server."""
    github_api: str = "https://api.github.com"
    github_raw: str = "https://raw.githubusercontent.com"
    s3_base: str = "https://s3.amazonaws.com/openneuro.org"

    @classmethod
    def from_env(cls) -> "Endpoints":
        env = os.environ
        return cls(
            github_api=env.get("OPENNEURO_EEG_TRIAGE_GITHUB_API", cls.github_api).rstrip("/"),
            github_raw=env.get("OPENNEURO_EEG_TRIAGE_GITHUB_RAW", cls.github_raw).rstrip("/"),
            s3_base=env.get("OPENNEURO_EEG_TRIAGE_S3_BASE", cls.s3_base).rstrip("/"),
        )


class HttpError(Exception):
    """An HTTP or network failure. ``status`` is the HTTP status code, or ``None`` for a network error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class Http:
    """GET helpers with retries on network errors and 5xx/429. 4xx (404, 403 ...) are raised immediately."""

    def __init__(self, timeout: float = 60.0, retries: int = 3, backoff: float = 1.0, token: str | None = None) -> None:
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")

    def _request(self, url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
        h = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if self.token and "api.github.com" in url:
            h["Authorization"] = f"Bearer {self.token}"
        if headers:
            h.update(headers)
        return urllib.request.Request(url, headers=h)

    def _open(self, url: str, headers: dict[str, str] | None = None):
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return urllib.request.urlopen(self._request(url, headers), timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    last = e
                else:
                    raise HttpError(f"HTTP {e.code} for {url}", e.code) from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
                if attempt >= self.retries:
                    raise HttpError(f"network error for {url}: {e}", None) from e
            time.sleep(self.backoff * (2 ** attempt))
        raise HttpError(f"failed for {url}: {last}", None)

    def get_bytes(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        with self._open(url, headers) as resp:
            return resp.read()

    def get_text(self, url: str) -> str:
        return self.get_bytes(url).decode("utf-8", errors="replace")

    def get_json(self, url: str) -> Any:
        try:
            return json.loads(self.get_bytes(url, {"Accept": "application/json"}))
        except json.JSONDecodeError as e:
            raise HttpError(f"invalid JSON from {url}: {e}", None) from e

    def download(self, url: str, dst: Path, chunk: int = 1 << 20) -> int:
        """Stream ``url`` to ``dst`` (through ``dst.part``, renamed when complete). Returns the number of bytes."""
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with self._open(url) as resp, open(part, "wb") as fh:
                    expected = resp.headers.get("Content-Length")
                    n = 0
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        fh.write(block)
                        n += len(block)
                if expected is not None and n != int(expected):
                    raise HttpError(f"incomplete download of {url}: {n} of {expected} bytes", None)
                part.replace(dst)
                return n
            except HttpError as e:
                last = e
                if e.status is not None and e.status < 500 and e.status != 429:
                    break
            except (OSError, TimeoutError, ConnectionError) as e:
                last = e
            finally:
                if part.exists():
                    try:
                        part.unlink()
                    except OSError:
                        pass
            if attempt < self.retries:
                time.sleep(self.backoff * (2 ** attempt))
        if isinstance(last, HttpError):
            raise last
        raise HttpError(f"download failed for {url}: {last}", None)
