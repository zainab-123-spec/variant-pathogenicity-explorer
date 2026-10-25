"""A tiny HTTP client with an on-disk cache, rate limiting and retries.

Why this module exists
----------------------
Public APIs (especially NCBI E-utilities) ask you not to hammer them:
NCBI allows ~3 requests/second without an API key and ~10/second with one.
We also do not want to re-download the same data every time we run the
pipeline. So every GET goes through :class:`CachedSession`, which

1. builds a cache key from the URL + parameters,
2. returns the cached file if it exists,
3. otherwise waits long enough to respect the per-host rate limit,
4. downloads, retries on transient errors (429/5xx) with back-off,
5. and saves the response to ``data/cache``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_CACHE_DIR = Path(os.environ.get("VPE_CACHE_DIR", "data/cache"))
USER_AGENT = "variant-pathogenicity-explorer/0.1 (educational project)"

#: Minimum seconds between two requests to the same host.
RATE_LIMITS: dict[str, float] = {
    "eutils.ncbi.nlm.nih.gov": 0.11 if os.environ.get("NCBI_API_KEY") else 0.34,
    "rest.uniprot.org": 0.1,
    "alphafold.ebi.ac.uk": 0.1,
}


class CachedSession:
    """GET requests with a disk cache, per-host rate limiting and retries."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        max_retries: int = 4,
        timeout: float = 60.0,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        self.offline = offline
        self._last_call: dict[str, float] = {}
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self.n_network_calls = 0

    # ------------------------------------------------------------------ cache
    def _key(self, url: str, params: Mapping[str, Any] | None) -> Path:
        # Never let secrets (API keys) become part of the cache key.
        clean = {k: v for k, v in sorted((params or {}).items()) if k != "api_key"}
        raw = url + "?" + json.dumps(clean, sort_keys=True, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
        host = urlparse(url).netloc.replace(".", "_")
        return self.cache_dir / host / f"{digest}.cache"

    # ------------------------------------------------------------ rate limit
    def _wait_for_slot(self, host: str) -> None:
        gap = RATE_LIMITS.get(host, 0.0)
        last = self._last_call.get(host)
        if last is not None:
            sleep_for = gap - (time.monotonic() - last)
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._last_call[host] = time.monotonic()

    # ------------------------------------------------------------------- GET
    def get_text(self, url: str, params: Mapping[str, Any] | None = None) -> str | None:
        """Return the response body as text, or ``None`` on HTTP 404.

        Raises ``RuntimeError`` if the request keeps failing after retries.
        """
        path = self._key(url, params)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return None if text == "__404__" else text
        if self.offline:
            raise RuntimeError(f"offline mode and no cache for {url}")

        host = urlparse(url).netloc
        req_params = dict(params or {})
        if host == "eutils.ncbi.nlm.nih.gov":
            req_params.setdefault("tool", "variant-pathogenicity-explorer")
            if os.environ.get("NCBI_EMAIL"):
                req_params.setdefault("email", os.environ["NCBI_EMAIL"])
            if os.environ.get("NCBI_API_KEY"):
                req_params.setdefault("api_key", os.environ["NCBI_API_KEY"])

        delay = 1.0
        last_error = ""
        for _attempt in range(self.max_retries + 1):
            self._wait_for_slot(host)
            try:
                self.n_network_calls += 1
                resp = self._session.get(url, params=req_params, timeout=self.timeout)
            except requests.RequestException as exc:  # network hiccup
                last_error = str(exc)
            else:
                if resp.status_code == 200:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(resp.text, encoding="utf-8")
                    return resp.text
                if resp.status_code == 404:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("__404__", encoding="utf-8")
                    return None
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(delay)
            delay *= 2
        raise RuntimeError(f"GET {url} failed: {last_error}")

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        """Like :meth:`get_text` but parses JSON (``None`` on 404)."""
        text = self.get_text(url, params)
        return None if text is None else json.loads(text)
