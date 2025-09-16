# dswxni/downloader.py
from __future__ import annotations

"""
Robust parallel downloader with optional checksum verification and resume.

Features
--------
- Parallel downloads via ThreadPoolExecutor
- Earthdata Login support (via env or ~/.netrc), redirects handled
- Atomic writes (.part -> final) and optional resume via HTTP Range
- Optional checksum verification (md5/sha1/sha256/sha512)
- Basic retries via urllib3 Retry on 429/5xx, plus request timeouts
- Skips existing files (or verifies and re-downloads on mismatch)
- Clean results as dataclasses for logging/pipelines

Example
-------
from dswxni.downloader import download_all, DownloadResult

urls = ["https://example.com/file1.h5", "https://example.com/file2.h5"]
results = download_all(urls, workers=8, verify={"https://.../file1.h5": ("md5","abcdef...")})
for r in results:
    print(r.status, r.url, "->", r.path)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple, Literal, Any
import hashlib
import logging
import os
import time
import netrc
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .settings import SETTINGS

log = logging.getLogger(__name__)


# ---------------------------------------------
# Data models
# ---------------------------------------------
Status = Literal["downloaded", "skipped", "exists", "verified", "failed"]

@dataclass
class DownloadResult:
    url: str
    path: Optional[Path]
    status: Status
    size: Optional[int]
    elapsed_s: Optional[float]
    checksum_ok: Optional[bool] = None
    error: Optional[str] = None
    http_status: Optional[int] = None
    resumed: Optional[bool] = None


# ---------------------------------------------
# Session / auth helpers
# ---------------------------------------------
DEFAULT_UA = "dswxni-downloader/0.1"

def _load_earthdata_creds() -> Optional[Tuple[str, str]]:
    """
    Prefer env variables from SETTINGS; else fall back to ~/.netrc for urs.earthdata.nasa.gov.
    """
    if SETTINGS.earthdata_username and SETTINGS.earthdata_password:
        return SETTINGS.earthdata_username, SETTINGS.earthdata_password

    try:
        nrc = netrc.netrc()  # ~/.netrc
        creds = nrc.authenticators("urs.earthdata.nasa.gov")
        if creds:
            # (login, account, password)
            return creds[0], creds[2]  # type: ignore[index]
    except FileNotFoundError:
        pass
    except Exception as e:
        log.debug("Failed to parse ~/.netrc: %s", e)

    return None


def _create_session(timeout: Optional[int] = None) -> requests.Session:
    """
    Create a configured requests.Session with retries and auth.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": DEFAULT_UA})

    retries = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=SETTINGS.parallel_downloads, pool_maxsize=SETTINGS.parallel_downloads)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # Earthdata auth: set Basic auth; servers will only use it when required.
    creds = _load_earthdata_creds()
    if creds:
        s.auth = creds

    # Stash timeout in session for convenience
    s.request_timeout = timeout or SETTINGS.timeout_s  # type: ignore[attr-defined]
    return s


# ---------------------------------------------
# Path helpers
# ---------------------------------------------
def _safe_filename_from_url(url: str) -> str:
    """
    Extract last path component from URL, ignoring query string.
    """
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path)
    return name or "download.bin"


def _unique_path(root: Path, name: str) -> Path:
    """
    Choose a non-colliding destination path under root by appending __N if necessary.
    """
    root.mkdir(parents=True, exist_ok=True)
    base, ext = os.path.splitext(name)
    path = root / name
    n = 1
    while path.exists():
        path = root / f"{base}__{n}{ext}"
        n += 1
    return path


def _target_path(url: str, root: Path, dedupe: bool) -> Path:
    """
    Compute a destination path under `root` for the given URL.
    """
    name = _safe_filename_from_url(url)
    return _unique_path(root, name) if dedupe else (root / name)


# ---------------------------------------------
# Checksums
# ---------------------------------------------
_HASHERS = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
}

def compute_checksum(path: Path, algo: str, chunk: int = 1024 * 1024) -> str:
    algo = algo.lower()
    if algo not in _HASHERS:
        raise ValueError(f"Unsupported checksum algo: {algo}")
    h = _HASHERS[algo]()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def verify_checksum(path: Path, algo: str, expected_hex: str) -> bool:
    try:
        got = compute_checksum(path, algo)
        return got.lower() == expected_hex.lower()
    except Exception as e:
        log.debug("Checksum error for %s: %s", path, e)
        return False


# ---------------------------------------------
# Core download
# ---------------------------------------------
CHUNK = 1024 * 1024  # 1 MiB

def _supports_range(session: requests.Session, url: str, timeout: int) -> bool:
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        return "bytes" in (r.headers.get("Accept-Ranges", "") or "").lower()
    except Exception:
        return False


def _download_one(
    session: requests.Session,
    url: str,
    outdir: Path,
    *,
    overwrite: bool = False,
    resume: bool = True,
    dedupe_names: bool = False,
    expected: Optional[Tuple[str, str]] = None,
    progress_bar: "tqdm | None" = None,
) -> DownloadResult:
    t0 = time.monotonic()
    http_status: Optional[int] = None
    checksum_ok: Optional[bool] = None
    resumed = False

    try:
        outdir = Path(outdir or SETTINGS.download_root)
        outdir.mkdir(parents=True, exist_ok=True)

        # Compute destination path
        dest = _target_path(url, outdir, dedupe=dedupe_names)
        tmp = dest.with_suffix(dest.suffix + ".part")

        # Skip if destination exists and no overwrite
        if dest.exists() and not overwrite:
            size = dest.stat().st_size
            # Optionally verify checksum if provided
            if expected:
                algo, hexval = expected
                checksum_ok = verify_checksum(dest, algo, hexval)
                if checksum_ok:
                    return DownloadResult(url, dest, "verified", size, 0.0, checksum_ok=True, http_status=None, resumed=False)
                else:
                    # Mismatch -> re-download
                    dest.unlink(missing_ok=True)

            else:
                return DownloadResult(url, dest, "exists", size, 0.0, checksum_ok=None, http_status=None, resumed=False)

        # Prepare resume headers if enabled and tmp exists and server supports range
        headers = {}
        mode = "wb"
        if resume and tmp.exists():
            if _supports_range(session, url, timeout=session.request_timeout):  # type: ignore[attr-defined]
                resume_from = tmp.stat().st_size
                headers["Range"] = f"bytes={resume_from}-"
                mode = "ab"
                resumed = True
            else:
                tmp.unlink(missing_ok=True)

        # Start request (GET)
        with session.get(url, stream=True, allow_redirects=True, headers=headers, timeout=session.request_timeout) as r:  # type: ignore[attr-defined]
            http_status = r.status_code

            # If we asked for Range and server ignored it (200), start over
            if resumed and r.status_code == 200:
                r.close()
                tmp.unlink(missing_ok=True)
                resumed = False
                with session.get(url, stream=True, allow_redirects=True, timeout=session.request_timeout) as r2:  # type: ignore[attr-defined]
                    http_status = r2.status_code
                    r2.raise_for_status()
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    with open(tmp, "wb") as f:
                        for chunk in r2.iter_content(chunk_size=CHUNK):
                            if chunk:
                                f.write(chunk)
                                if progress_bar is not None:
                                    progress_bar.update(len(chunk))
            else:
                r.raise_for_status()
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, mode) as f:
                        for chunk in r.iter_content(chunk_size=CHUNK):
                            if chunk:
                                f.write(chunk)
                                if progress_bar is not None:
                                    progress_bar.update(len(chunk))
        # Atomic replace
        tmp.replace(dest)

        size = dest.stat().st_size

        # Optional checksum verification
        if expected:
            algo, hexval = expected
            checksum_ok = verify_checksum(dest, algo, hexval)
            status: Status = "downloaded" if checksum_ok or checksum_ok is None else "failed"
            # If checksum failed, mark failed (do not delete automatically)
            elapsed = time.monotonic() - t0
            return DownloadResult(
                url, dest, "downloaded" if checksum_ok else "failed", size, elapsed,
                checksum_ok=checksum_ok, http_status=http_status, resumed=resumed,
                error=None if checksum_ok else f"checksum mismatch ({algo})"
            )

        elapsed = time.monotonic() - t0
        return DownloadResult(url, dest, "downloaded", size, elapsed, checksum_ok=None, http_status=http_status, resumed=resumed)

    except Exception as e:
        elapsed = time.monotonic() - t0
        return DownloadResult(url, None, "failed", None, elapsed, checksum_ok=None, error=str(e), http_status=http_status, resumed=resumed)


# ---------------------------------------------
# Public API
# ---------------------------------------------
def download_all(
    urls: Iterable[str],
    outdir: Optional[Path | str] = None,
    *,
    workers: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = True,
    dedupe_names: bool = False,
    verify: Optional[Mapping[str, Tuple[str, str]]] = None,
    show_progress: Literal["bytes", "files", "none"] = "bytes",
) -> list[DownloadResult]:
    """
    Download multiple URLs in parallel with retries and optional checksum verification.

    Parameters
    ----------
    urls : Iterable[str]
        HTTP(S) URLs to fetch.
    outdir : Path | str, optional
        Destination root directory (default: SETTINGS.download_root).
    workers : int, optional
        Number of parallel workers (default: SETTINGS.parallel_downloads).
    overwrite : bool
        If True, re-download even if the file exists.
    resume : bool
        If True, attempt to resume partial downloads (.part) when server supports Range.
    dedupe_names : bool
        If True, avoid name collisions by appending __N to filenames.
    verify : Mapping[str, (algo, hexdigest)], optional
        Map of url -> (checksum algorithm, expected hex). If provided, files are
        verified after download. Existing files are verified and marked "verified"
        on match; on mismatch, they are re-downloaded.

    Returns
    -------
    list[DownloadResult]
    """
    outdir = Path(outdir or SETTINGS.download_root)
    outdir.mkdir(parents=True, exist_ok=True)
    workers = workers or SETTINGS.parallel_downloads
    verify = verify or {}

    session = _create_session()

    bytes_bar: tqdm | None = None
    if show_progress == "bytes":
        # best-effort total size
        size_map = head_sizes(urls)  # existing helper
        total = sum(s for s in size_map.values() if isinstance(s, int))
        # If all unknown, total=0 → tqdm will behave as indeterminate; still okay.
        bytes_bar = tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
            leave=True,
            dynamic_ncols=True,
        )

    results: list[DownloadResult] = []
    fut_to_url: dict[Any, str] = {}


    with ThreadPoolExecutor(max_workers=workers) as ex:
        for url in urls:
            expected = verify.get(url)
            fut = ex.submit(
                _download_one,
                session,
                url,
                outdir,
                overwrite=overwrite,
                resume=resume,
                dedupe_names=dedupe_names,
                expected=expected,
                progress_bar=bytes_bar,   # <- pass shared bar (may be None)
            )
            fut_to_url[fut] = url

        iterator = as_completed(fut_to_url)
        if show_progress == "files":
            iterator = tqdm(iterator, total=len(fut_to_url), desc="Files", leave=True, dynamic_ncols=True)

        for fut in iterator:
            res = fut.result()
            results.append(res)
            if res.status in ("downloaded", "verified", "exists"):
                log.info("%s: %s -> %s (%s)", res.status, res.url, res.path, f"{res.size} B" if res.size is not None else "?")
            else:
                log.warning("%s: %s (%s)", res.status, res.url, res.error or "unknown error")

    # Close bars
    if isinstance(iterator, tqdm):
        iterator.close()
    if bytes_bar is not None:
        bytes_bar.close()

    return results


def head_sizes(urls: Iterable[str]) -> dict[str, Optional[int]]:
    """
    Issue HEAD requests to retrieve Content-Length (if present).

    Returns mapping url -> size or None.
    """
    session = _create_session()
    out: dict[str, Optional[int]] = {}
    for url in urls:
        try:
            r = session.head(url, allow_redirects=True, timeout=session.request_timeout)  # type: ignore[attr-defined]
            size_str = r.headers.get("Content-Length")
            out[url] = int(size_str) if size_str is not None else None
        except Exception as e:
            log.debug("HEAD failed for %s: %s", url, e)
            out[url] = None
    return out
