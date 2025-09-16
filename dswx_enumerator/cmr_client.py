# dswxni/cmr_client.py
from __future__ import annotations

import dataclasses as dc
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Iterable, Literal
import json
import logging
import os
import time

import requests

__all__ = [
    "Granule",
    "Link",
    "CollectionSpec",
    "Spatial",
    "Temporal",
    "ExtraFilters",
    "search_cmr_flexible",
    "search_cmr",
    "resolve_collection_concept_id",
]

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
CMR_COLLECTIONS = "https://cmr.earthdata.nasa.gov/search/collections.json"

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "dswxni/0.1 (+https://example.com)"
}
DEFAULT_TIMEOUT = 60  # seconds
MAX_PAGE_SIZE = 2000

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Known CID registry (can be extended)
# Key format: (provider_upper, short_name)
KNOWN_CIDS: dict[tuple[str, str], str] = {
    ("ASF", "ALOS_PALSAR_RTC_HiRes"): "C1206487504-ASF",  # ALOS-1 High-Res Terrain Corrected
    # Add more if you like:
    # ("POCLOUD", "NISAR_L2_GCOV"): "Cxxxxxxxxxx-POCLOUD",
}

# Environment override: DSWXNI_CMR_SHORTNAME_MAP='{"ASF:ALOS_PALSAR_RTC_HiRes":"C1206487504-ASF"}'
ENV_SHORTNAME_MAP = "DSWXNI_CMR_SHORTNAME_MAP"


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
@dc.dataclass
class Link:
    href: str
    rel: str | None = None
    title: str | None = None
    type: str | None = None
    inherited: bool | None = None


@dc.dataclass
class Granule:
    """
    Provider-agnostic granule record normalized from CMR JSON.
    """
    id: str
    title: str
    collection: str | None
    producer_granule_id: str | None
    time_start: str | None
    time_end: str | None
    size_mb: float | None
    links: list[Link]
    # Useful extras (present when available)
    provider: str | None = None
    concept_id: str | None = None
    orbit: str | None = None
    checksum: str | None = None
    checksum_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["links"] = [asdict(l) for l in self.links]
        return d

    def pick_best_download(
        self,
        *,
        provider_hint: str | None = None,
        allow_opendap: bool = False,
        prefer_ext: tuple[str, ...] | None = None,
    ) -> str | None:
        """
        Pick a primary data URL with sensible defaults per provider.

        - Prefer HTTPS (non-OPeNDAP) direct data links.
        - If provider looks like ASF → prefer archives (.zip/.tar.*) for SAR scenes.
        - If provider looks like PO.DAAC → prefer HDF5/NetCDF/GeoTIFF HTTP(S) data.
        - Fall back to first HTTP(S) link; only use OPeNDAP when allowed.
        """
        hrefs = [l.href for l in self.links if l.href and l.href.startswith(("http://", "https://"))]
        if not hrefs:
            return None

        prov = (provider_hint or (self.provider or "")).lower()

        if prefer_ext is None:
            if "asf" in prov:
                prefer_ext = (".zip", ".tar.gz", ".tgz", ".tar")
            else:  # PO.DAAC or other
                prefer_ext = (".h5", ".nc", ".tif", ".tiff", ".zip")

        def is_http_data(h: str) -> bool:
            return h.startswith("https://") and ("/opendap" not in h)

        for h in hrefs:
            if is_http_data(h) and h.lower().endswith(prefer_ext):
                return h
        for h in hrefs:
            if is_http_data(h):
                return h
        if allow_opendap:
            for h in hrefs:
                if "/opendap" in h:
                    return h
        return hrefs[0]


@dc.dataclass
class CollectionSpec:
    """
    Identifies a collection. Use concept_id when available (unambiguous).
    If concept_id is not provided, we can auto-resolve from short_name.
    """
    short_name: str | None = None   # e.g., "ALOS_PALSAR_RTC_HiRes", "NISAR_L2_GCOV"
    concept_id: str | None = None   # e.g., "C1206487504-ASF"
    version: str | None = None      # e.g., "003"
    provider: str | None = None     # e.g., "ASF", "POCLOUD"
    data_center: str | None = None  # rarely needed


@dc.dataclass
class Spatial:
    bbox: tuple[float, float, float, float] | None = None  # lon_min, lat_min, lon_max, lat_max
    polygon_wkt: str | None = None                         # WKT polygon (lon/lat)


@dc.dataclass
class Temporal:
    start: str | None = None  # ISO8601
    end: str | None = None    # ISO8601


@dc.dataclass
class ExtraFilters:
    platform: str | None = None
    instrument: str | None = None
    processing_level: str | None = None
    day_night_flag: str | None = None
    orbit_direction: Literal["ASCENDING", "DESCENDING", "ascending", "descending"] | None = None
    additional: dict[str, Any] | None = None  # pass-through raw CMR params


# ---------------------------------------------------------------------
# HTTP helpers with basic retry
# ---------------------------------------------------------------------
def _cmr_get(url: str, params: dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    backoff = 1.0
    for attempt in range(5):
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code == 429 or 500 <= r.status_code < 600:
            log.warning("CMR %s returned %s; retrying in %.1fs (attempt %d)",
                        url, r.status_code, backoff, attempt + 1)
            time.sleep(backoff)
            backoff = min(backoff * 2, 16.0)
            continue
        r.raise_for_status()
        return r.json()
    r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------
# Concept-id resolver
# ---------------------------------------------------------------------
def _load_env_shortname_map() -> dict[str, str]:
    raw = os.environ.get(ENV_SHORTNAME_MAP, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        log.warning("Failed to parse %s: %s", ENV_SHORTNAME_MAP, e)
    return {}

@lru_cache(maxsize=256)
def resolve_collection_concept_id(
    short_name: str | None,
    *,
    provider: str | None = None,
    version: str | None = None,
) -> str | None:
    """
    Resolve a collection concept-id from short_name (+ optional provider/version).

    Resolution order:
    1) Environment override map (DSWXNI_CMR_SHORTNAME_MAP) with keys like:
       - "ASF:ALOS_PALSAR_RTC_HiRes"
       - "ASF:ALOS_PALSAR_RTC_HiRes:003"
    2) Built-in KNOWN_CIDS dictionary.
    3) Live query to CMR collections.json filtered by short_name/provider/version.

    Returns None if not found.
    """
    if not short_name:
        return None

    prov_key = (provider or "").upper()
    env_map = _load_env_shortname_map()
    # Try version-qualified first
    if version:
        k = f"{prov_key}:{short_name}:{version}"
        if k in env_map:
            return env_map[k]
    # Then unqualified
    k = f"{prov_key}:{short_name}"
    if k in env_map:
        return env_map[k]

    # Built-in registry
    if (prov_key, short_name) in KNOWN_CIDS:
        return KNOWN_CIDS[(prov_key, short_name)]

    # Query CMR collections endpoint
    params: dict[str, Any] = {"short_name": short_name, "page_size": 2000}
    if provider:
        params["provider"] = provider
    if version:
        params["version"] = version

    payload = _cmr_get(CMR_COLLECTIONS, params)
    entries = (payload.get("feed", {}).get("entry", []) or [])
    if not entries:
        return None

    # If multiple, prefer exact provider match and any 'has_granules' flag
    def score(e: dict[str, Any]) -> tuple[int, int]:
        prov_match = 1 if (not provider or e.get("archive_center") == provider or e.get("data_center") == provider) else 0
        has_granules = 1 if e.get("has_granules") else 0
        return (prov_match, has_granules)

    entries.sort(key=score, reverse=True)
    cid = entries[0].get("id") or entries[0].get("concept_id")
    return cid


# ---------------------------------------------------------------------
# Param builder / parsing
# ---------------------------------------------------------------------
def _parse_links(entry: dict[str, Any]) -> list[Link]:
    links = []
    for l in entry.get("links", []) or []:
        href = l.get("href")
        if not href:
            continue
        links.append(
            Link(
                href=href,
                rel=l.get("rel"),
                title=l.get("title"),
                type=l.get("type"),
                inherited=l.get("inherited"),
            )
        )
    return links


def _entry_to_granule(e: dict[str, Any]) -> Granule:
    size_mb = float(e.get("granule_size", 0.0)) if e.get("granule_size") else None

    checksum = e.get("checksum_value") or e.get("data_checksum") or None
    checksum_type = e.get("checksum_algorithm") or e.get("checksum_type") or None

    return Granule(
        id=e["id"],
        producer_granule_id=e.get("producer_granule_id"),
        title=e.get("title", e["id"]),
        collection=e.get("dataset_id"),
        time_start=e.get("time_start"),
        time_end=e.get("time_end"),
        size_mb=size_mb,
        links=_parse_links(e),
        provider=(e.get("data_center") or e.get("archive_center")),
        concept_id=e.get("collection_concept_id"),
        orbit=e.get("orbit_number") or e.get("orbit") or e.get("orbit_calculated_spatial_domains")[0].get("orbit_number"),
        checksum=checksum,
        checksum_type=checksum_type,
    )


def _build_params(
    spec: CollectionSpec,
    spatial: Spatial | None,
    temporal: Temporal | None,
    extra: ExtraFilters | None,
    page_size: int,
    *,
    auto_resolve_cid: bool,
) -> dict[str, Any]:
    p: dict[str, Any] = {"page_size": min(max(page_size, 1), MAX_PAGE_SIZE)}

    # Auto-resolve concept_id from short_name if asked
    concept_id = spec.concept_id
    if auto_resolve_cid and not concept_id and spec.short_name:
        concept_id = resolve_collection_concept_id(
            spec.short_name, provider=spec.provider, version=spec.version
        )

    if concept_id:
        p["collection_concept_id"] = concept_id
    else:
        # Fall back to short_name filters if no CID
        if spec.short_name:
            p["short_name"] = spec.short_name
        if spec.version:
            p["version"] = spec.version
        if spec.provider:
            p["provider"] = spec.provider
        if spec.data_center:
            p["data_center_h"] = spec.data_center  # rarely used

    # Spatial
    if spatial and spatial.bbox:
        p["bounding_box"] = ",".join(map(str, spatial.bbox))
    if spatial and spatial.polygon_wkt:
        p["polygon"] = spatial.polygon_wkt

    # Temporal
    if temporal and (temporal.start or temporal.end):
        p["temporal"] = f"{temporal.start or ''},{temporal.end or ''}"

    # Extras
    if extra:
        if extra.platform:
            p["platform"] = extra.platform
        if extra.instrument:
            p["instrument"] = extra.instrument
        if extra.processing_level:
            p["processing_level"] = extra.processing_level
        if extra.day_night_flag:
            p["day_night_flag"] = extra.day_night_flag
        if extra.orbit_direction:
            p["orbit_direction"] = extra.orbit_direction
        if extra.additional:
            p.update(extra.additional)

    return p


# ---------------------------------------------------------------------
# Public search APIs
# ---------------------------------------------------------------------
def search_cmr_flexible(
    spec: CollectionSpec,
    *,
    spatial: Spatial | None = None,
    temporal: Temporal | None = None,
    extra: ExtraFilters | None = None,
    max_items: int = 200,
    page_size: int = 200,
    auto_resolve_cid: bool = True,
) -> list[Granule]:
    """
    Rich CMR search with automatic pagination and normalized outputs.

    If `concept_id` is not provided, will try to auto-resolve from short_name
    (using provider/version hints, built-ins, env map, or a collections API query).
    """
    params = _build_params(
        spec, spatial, temporal, extra, page_size, auto_resolve_cid=auto_resolve_cid
    )
    out: list[Granule] = []
    page_num = 1

    while len(out) < max_items:
        params_page = dict(params)
        params_page["page_num"] = page_num
        payload = _cmr_get(CMR_GRANULES, params_page)
        entries = (payload.get("feed", {}).get("entry", []) or [])
        if not entries:
            break

        for aef, e in enumerate(entries):
            if aef == 0:
                print(e)
            out.append(_entry_to_granule(e))
            if len(out) >= max_items:
                break

        page_num += 1

    return out


def search_cmr(
    short_name: str,
    bbox: tuple[float, float, float, float] | None,
    start: str | None,
    end: str | None,
    max_items: int = 100,
    *,
    provider: str | None = None,
    version: str | None = None,
    extra_params: dict[str, Any] | None = None,
    auto_resolve_cid: bool = True,
) -> list[Granule]:
    """
    Backward-compatible, simpler wrapper.

    If `auto_resolve_cid` is True (default), the function will resolve the
    collection concept-id first and query by CID for speed/precision.
    """
    spec = CollectionSpec(short_name=short_name, provider=provider, version=version)
    spatial = Spatial(bbox=bbox) if bbox else None
    temporal = Temporal(start=start, end=end)
    extra = ExtraFilters(additional=extra_params)
    return search_cmr_flexible(
        spec, spatial=spatial, temporal=temporal, extra=extra,
        max_items=max_items, auto_resolve_cid=auto_resolve_cid
    )
