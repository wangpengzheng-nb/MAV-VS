from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .models import ResearchSourceResult


class ResearchHttpClient:
    """HTTP client with bounded retries, diagnostics and per-task raw snapshots."""

    def __init__(self, snapshot_dir: Path | None = None, *, attempts: int = 3,
                 connect_timeout: float = 5.0, read_timeout: float = 20.0):
        self.snapshot_dir = snapshot_dir
        self.attempts = attempts
        self.timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self._last_request: dict[str, float] = {}
        self._rate_lock = threading.Lock()
        if snapshot_dir:
            snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self, host: str, min_interval: float) -> None:
        if min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = min_interval - (now - self._last_request.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()

    def _snapshot(self, source: str, url: str, attempt: int, response: httpx.Response) -> str:
        if not self.snapshot_dir:
            return ""
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        path = self.snapshot_dir / f"{source}_{digest}_{attempt}.json"
        payload = {
            "url": url, "status_code": response.status_code,
            "headers": {k: v for k, v in response.headers.items()
                       if k.lower() in {"content-type", "etag", "last-modified", "retry-after"}},
            "body": response.text,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def get_json(self, source: str, url: str, *, params: dict[str, Any] | None = None,
                 min_interval: float = 0.0) -> tuple[Any | None, ResearchSourceResult]:
        started = time.monotonic()
        snapshots: list[str] = []
        last_error = ""
        last_type = ""
        host = urlparse(url).netloc
        headers = {"User-Agent": "AutoVS-Agent/0.2 target-research"}
        with httpx.Client(timeout=self.timeout, follow_redirects=False, headers=headers) as client:
            for attempt in range(1, self.attempts + 1):
                self._throttle(host, min_interval)
                try:
                    response = client.get(url, params=params)
                    snap = self._snapshot(source, str(response.request.url), attempt, response)
                    if snap:
                        snapshots.append(snap)
                    if response.status_code == 429 or response.status_code >= 500:
                        last_type = f"http_{response.status_code}"
                        last_error = response.text[:300]
                        if attempt < self.attempts:
                            retry_after = response.headers.get("Retry-After", "")
                            try:
                                delay = min(float(retry_after), 10.0)
                            except ValueError:
                                delay = min(0.5 * (2 ** (attempt - 1)) + random.random() * 0.2, 3.0)
                            time.sleep(delay)
                            continue
                    if 400 <= response.status_code < 500:
                        return None, ResearchSourceResult(
                            source=source, status="invalid", attempts=attempt,
                            latency_ms=int((time.monotonic() - started) * 1000),
                            error_type=f"http_{response.status_code}",
                            message=response.text[:500], snapshot_paths=snapshots,
                        )
                    response.raise_for_status()
                    try:
                        data = response.json()
                    except (ValueError, json.JSONDecodeError) as exc:
                        return None, ResearchSourceResult(
                            source=source, status="invalid", attempts=attempt,
                            latency_ms=int((time.monotonic() - started) * 1000),
                            error_type="invalid_json", message=str(exc), snapshot_paths=snapshots,
                        )
                    empty = data in (None, {}, [])
                    return data, ResearchSourceResult(
                        source=source, status="empty" if empty else "success", attempts=attempt,
                        latency_ms=int((time.monotonic() - started) * 1000), snapshot_paths=snapshots,
                    )
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                    last_type = type(exc).__name__
                    last_error = str(exc)
                    if attempt < self.attempts:
                        time.sleep(min(0.5 * (2 ** (attempt - 1)) + random.random() * 0.2, 3.0))
        return None, ResearchSourceResult(
            source=source, status="unavailable", attempts=self.attempts,
            latency_ms=int((time.monotonic() - started) * 1000),
            error_type=last_type, message=last_error[:500], snapshot_paths=snapshots,
        )
