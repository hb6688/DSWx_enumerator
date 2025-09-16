# dswx_enumerator/cli.py
from __future__ import annotations

"""
Command Line Interface for DSWx-enumerator utilities

This CLI exposes three primary workflows:

1) search      — Query NASA CMR for granules (ASF / PO.DAAC, etc.)
2) download    — Download the matched granules to your local cache
3) build-yaml  — Render runconfig YAML files (one per granule)

Run `dswx_enumerator COMMAND --help` for detailed instructions and examples.
"""

import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional
import re
from collections import defaultdict

import click

from .cmr_client import (
    search_cmr_flexible,
    resolve_collection_concept_id,
    CollectionSpec,
    Spatial,
    Temporal,
)
from .mgrs_db_reader import _polygons_from_db_general
from .downloader import download_all
from .settings import SETTINGS
from .yaml_builder import render_runconfig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_bbox(bbox_str: Optional[str]) -> tuple[float, float, float, float] | None:
    """
    Parse "minx,miny,maxx,maxy" into a 4-tuple of floats, or None.
    """
    if not bbox_str:
        return None
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise click.BadParameter("Expected 4 comma-separated numbers: minx,miny,maxx,maxy")
    try:
        xmin, ymin, xmax, ymax = map(float, parts)
    except ValueError:
        raise click.BadParameter("BBox values must be numeric (lon/lat)")
    if xmin >= xmax or ymin >= ymax:
        raise click.BadParameter("BBox must satisfy minx < maxx and miny < maxy")
    return (xmin, ymin, xmax, ymax)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise click.ClickException(f"Failed to read JSON from {path}: {e}")

_TRACK_PATTERNS = [
    re.compile(r"\bT0*(\d{3,5})\b", re.IGNORECASE),      # T00777 or T1234
    re.compile(r"\bTrack[:\s]*0*(\d{1,5})\b", re.IGNORECASE),
]

def _extract_track(granule_dict: dict) -> str | None:
    # Prefer explicit fields if present
    for k in ("orbit", "orbit_number", "track", "relative_orbit", "relativeOrbit", "path"):
        v = granule_dict.get(k)
        if isinstance(v, (int, float)) and v >= 0:
            return str(int(v))
        if isinstance(v, str) and v.strip():
            # If it's already a number-like string, use it
            if re.fullmatch(r"\d{1,5}", v.strip()):
                return v.strip()
            # Maybe embedded like "..._T00777_..."
            for pat in _TRACK_PATTERNS:
                m = pat.search(v)
                if m:
                    return str(int(m.group(1)))  # drop leading zeros

    # Fall back to parsing the title / id
    for field in ("title", "id"):
        s = granule_dict.get(field)
        if isinstance(s, str):
            for pat in _TRACK_PATTERNS:
                m = pat.search(s)
                if m:
                    return str(int(m.group(1)))
    return None

def _extract_track_s1(granule_dict: dict) -> str | None:
    producer_granule_id = granule_dict.get("producer_granule_id")
    print(producer_granule_id)
    if producer_granule_id:
        return producer_granule_id.split("_")[3].split("-")[0][1:]
    return None

def _default_ni_template_path() -> Path:
    """
    Return <repo_root>/templates/runconfig_dswx_ni.j2.
    Assumes this file lives at <repo_root>/dswx_enumerator/cli.py
    """
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "dswx_enumerator" / "template" / "runconfig_dswx_ni.j2"

# -----------------------------------------------------------------------------
# Top-level group
# -----------------------------------------------------------------------------
@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,  # wider help
    }
)
@click.version_option(package_name="dswx_enumerator")
def main():
    """
    DSWx-enumerator Toolkit — search, download, and prepare runconfig YAMLs.

    Environment variables (optional):

\b
      - DSWX_DOWNLOAD_ROOT        Override default download root
      - DSWX_PARALLEL_DOWNLOADS   Number of downloader threads (default: 6)
      - DSWX_TIMEOUT_S            HTTP timeout in seconds (default: 45)
      - DSWX_EARTHDATA_USERNAME   Earthdata username (fallback to ~/.netrc)
      - DSWX_EARTHDATA_PASSWORD   Earthdata password
      - DSWX_CMR_SHORTNAME_MAP    JSON map "PROVIDER:SHORT[:VERSION]":"C1234-DC"

    Tip:

\b
      Prefer `concept_id` for unambiguous collection selection. If you only
      know a short name, we’ll try to auto-resolve to the right Concept-ID.
    """


# -----------------------------------------------------------------------------
# search
# -----------------------------------------------------------------------------
@main.command("search")
@click.option(
    "--short-name",
    help=(
        "CMR collection short name (e.g., 'ALOS_PALSAR_RTC_HiRes', 'NISAR_L2_GCOV'). "
        "If provided without --concept-id, the CLI will attempt to auto-resolve the "
        "collection Concept-ID using provider/version hints, a small built-in map, "
        "and a live CMR collections query."
    ),
)
@click.option(
    "--concept-id",
    help=(
        "CMR collection Concept-ID (e.g., 'C1206487504-ASF'). "
        "Best for speed and precision; overrides --short-name if both are given."
    ),
)
@click.option(
    "--track",
    help=(
        "Optional track number to select (e.g., '777'). "
        "If provided, only granules from this track are kept and results are saved. "
        "If omitted, results are summarized to stdout (not saved), and a warning is shown "
        "if multiple tracks are present."
    ),
)
@click.option(
    "--provider",
    help=(
        "Data center/provider hint (e.g., 'ASF', 'POCLOUD'). "
        "Used for concept-id resolution and as a search filter when --concept-id "
        "is not provided."
    ),
)
@click.option(
    "--version",
    help=(
        "Collection version (e.g., '003'). Used as a hint when resolving Concept-ID "
        "from --short-name."
    ),
)
@click.option(
    "--bbox",
    help=(
        "Search bounding box as 'minlon,minlat,maxlon,maxlat' (WGS84). "
        "Example: --bbox '-124,45,-121.5,46.5'."
    ),
)
@click.option(
    "--polygon",
    help=(
        "Search polygon in WKT (lon/lat). Use either --bbox or --polygon (not both). "
        "Example: \"POLYGON((-122.6 37.1,-122.6 37.7,-121.9 37.7,-121.9 37.1,-122.6 37.1))\""
    ),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    help=(
        "Path to local SQLite database that stores polygons. "
        "Only required if using --mgrs-set-id."
    ),
)
@click.option(
    "--mgrs-set-id",
    help=(
        "MGRS set identifier to look up in --db. "
        "If provided (with --db), overrides --bbox/--polygon."
    ),
)
@click.option(
    "--mgrs-table",
    default="mgrs_track_frame_db",
    show_default=True,
    help=(
        "Table name in --db containing polygons (plain-WKT mode)."
    ),
)
@click.option(
    "--mgrs-id-col",
    default="mgrs_set_id",
    show_default=True,
    help=(
        "ID column name in --db that matches --mgrs-set-id."
    ),
)
@click.option(
    "--start",
    help="Start time (ISO8601). Example: 2007-01-01T00:00:00Z",
)
@click.option(
    "--end",
    help="End time (ISO8601). Example: 2011-12-31T23:59:59Z",
)
@click.option(
    "--max",
    "max_items",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of granules to return (across pages).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSON file to write the normalized search results.",
)
def search_cmd(
    short_name: Optional[str],
    concept_id: Optional[str],
    provider: Optional[str],
    version: Optional[str],
    bbox: Optional[str],
    polygon: Optional[str],
    db_path: Optional[Path],
    mgrs_set_id: Optional[str],
    mgrs_table: str,
    mgrs_id_col: str,
    track: Optional[str],
    start: Optional[str],
    end: Optional[str],
    max_items: int,
    out_path: Path,
):
    """
    Search CMR for granules and save results to JSON.

    Modes:
      • BBox/Polygon (no DB): provide --bbox or --polygon
      • MGRS via DB: provide BOTH --db and --mgrs-set-id
    """
    # identity
    if not concept_id and not short_name:
        raise click.UsageError(
            "Provide either --concept-id or --short-name (with optional --provider/--version)."
        )

    using_db = bool(db_path and mgrs_set_id)
    using_bbox_poly = bool(bbox or polygon)

    if using_db and using_bbox_poly:
        raise click.BadParameter(
            "Choose ONE spatial mode: either --db + --mgrs-set-id OR --bbox/--polygon."
        )
    if not using_db and not using_bbox_poly:
        raise click.BadParameter(
            "Provide spatial constraints: either --bbox/--polygon OR --db + --mgrs-set-id."
        )

    # resolve concept id if needed
    if not concept_id and short_name:
        concept_id = resolve_collection_concept_id(short_name, provider=provider, version=version)

    spec = CollectionSpec(short_name=short_name, concept_id=concept_id, provider=provider, version=version)

    # spatial selection
    if using_db:
        cmr_polys = _polygons_from_db_general(
            db_path=db_path,
            mgrs_set_id=mgrs_set_id,
            layer=mgrs_table,         # same option name as before; here used as GPKG/SpatiaLite layer
            id_col=mgrs_id_col,
        )
        click.echo(f"Using polygon from DB for mgrs_set_id={mgrs_set_id}")
        # Query CMR per polygon and merge results by granule id
        by_id: dict[str, dict] = {}
        for i, poly_str in enumerate(cmr_polys, 1):
            spatial = Spatial(polygon_wkt=poly_str)
            part_granules = search_cmr_flexible(
                spec,
                spatial=spatial,
                temporal=Temporal(start=start, end=end),
                max_items=max_items,
            )
            for g in part_granules:
                by_id[g.id] = g.to_dict()
            click.echo(f"  part {i}: {len(part_granules)} granules")

        payload = list(by_id.values())

    else:
        # BBox/Polygon mode (no DB)
        if bbox and polygon:
            raise click.BadParameter("Use either --bbox or --polygon, not both.")
        spatial = Spatial(polygon_wkt=polygon) if polygon else Spatial(bbox=_parse_bbox(bbox))

        granules = search_cmr_flexible(
            spec,
            spatial=spatial,
            temporal=Temporal(start=start, end=end),
            max_items=max_items,
        )
        payload = [g.to_dict() for g in granules]

    by_track: dict[str, list[dict]] = defaultdict(list)
    unknown_key = "unknown"
    for d in payload:
        if short_name == "OPERA_L2_RTC-S1_V1" or concept_id == "C2777436413-ASF":
            t = _extract_track_s1(d) # producer_granule_id
        else:
            t = _extract_track(d) or unknown_key
        by_track[t].append(d)

    if track:
        # Filter to the requested track
        kept = by_track.get(track) or []
        if not kept:
            click.echo(f"No granules found for track={track}. Available tracks: {', '.join(sorted(k for k in by_track.keys() if k != unknown_key)) or '(none)'}")
        # Save filtered (even if empty, per your spec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(kept, indent=2))
        click.echo(f"Saved {len(kept)} results for track={track} → {out_path}")
        return

    # No --track provided → print summary and do NOT save
    tracks_sorted = sorted(by_track.keys(), key=lambda k: (k==unknown_key, k))
    total = sum(len(v) for v in by_track.values())
    multi = len([k for k in by_track.keys() if k != unknown_key]) > 1

    if multi:
        click.echo("⚠️  Multiple tracks detected; this may be unexpected.\n")

    click.echo(f"Total granules: {total}")
    for k in tracks_sorted:
        rows = by_track[k]
        click.echo(f"- track={k} : {len(rows)}")
        # show a few sample titles
        for trow in rows[:3]:
            click.echo(f"    • {trow.get('title') or trow.get('id')}")

    click.echo("\nTip: re-run with --track <number> to select and save only that track.")
    click.echo(f"      e.g., --track {next((k for k in tracks_sorted if k != unknown_key), '777')} --out {out_path}")
    return
    # write output once
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # out_path.write_text(json.dumps(payload, indent=2))
    # click.echo(f"Saved {len(payload)} results → {out_path}")


# -----------------------------------------------------------------------------
# download
# -----------------------------------------------------------------------------
@main.command("download")
@click.argument("search_json", type=click.Path(path_type=Path))
@click.option(
    "--outdir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Destination directory for downloads. "
        "Default is DSWX_DOWNLOAD_ROOT (currently: "
        f"{SETTINGS.download_root})."
    ),
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help=(
        "Number of parallel download workers. "
        f"Default is DSWX_PARALLEL_DOWNLOADS (currently: {SETTINGS.parallel_downloads})."
    ),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Re-download even if the destination file already exists.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Attempt to resume partial downloads when the server supports HTTP Range.",
)
@click.option(
    "--dedupe-names/--no-dedupe-names",
    default=False,
    show_default=True,
    help="Append '__N' to filenames to avoid collisions instead of overwriting.",
)
def download_cmd(
    search_json: Path,
    outdir: Optional[Path],
    workers: Optional[int],
    overwrite: bool,
    resume: bool,
    dedupe_names: bool,
):
    """
    Download all URLs from a search results JSON.

    The JSON is the output produced by `dswx_enumerator search` (a list of normalized
    granule dicts). We extract HTTP(S) links and fetch them in parallel.

    Notes:
      • Earthdata Login credentials can come from ~/.netrc (preferred) or env vars.
      • Existing files are skipped unless --overwrite is set.
      • With --resume, partial '.part' files continue if the server supports it.
    """
    ALLOWED_EXT = {".zip", ".tif", ".tiff", ".h5"}  # tweak if you want more

    data = _read_json(search_json)

    # Prefer strict HTTPS data links (avoid OPeNDAP), fall back to first HTTP(S).
    def _is_http(u: str) -> bool:
        try:
            scheme = urlparse(u).scheme.lower()
            return scheme in ("http", "https")
        except Exception:
            return False

    def _has_allowed_ext(u: str) -> bool:
        try:
            path = urlparse(u).path  # strips query/fragment
            ext = Path(path).suffix.lower()
            return ext in ALLOWED_EXT
        except Exception:
            return False

    data = _read_json(search_json)

    # Prefer HTTPS direct data links with allowed extensions.
    urls = []
    for d in data:
        links = d.get("links", []) or []

        # 1) best: https + allowed extension, and not OPeNDAP
        best = [
            l["href"] for l in links
            if isinstance(l, dict)
            and _is_http(str(l.get("href", "")))
            and _has_allowed_ext(str(l.get("href", "")))
            and "/opendap" not in str(l.get("href", "")).lower()
            and str(l.get("href", "")).startswith("https://")
        ]

        # 2) fallback: http(s) + allowed extension (any scheme), still avoid OPeNDAP
        if not best:
            best = [
                l["href"] for l in links
                if isinstance(l, dict)
                and _is_http(str(l.get("href", "")))
                and _has_allowed_ext(str(l.get("href", "")))
                and "/opendap" not in str(l.get("href", "")).lower()
            ]

        # 3) last resort: first http(s) link (in case provider uses uncommon names)
        if not best:
            best = [
                l["href"] for l in links
                if isinstance(l, dict)
                and _is_http(str(l.get("href", "")))
                and "/opendap" not in str(l.get("href", "")).lower()
            ]

        if best:
            urls.append(best[0])

    # De-duplicate while preserving order
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    if not urls:
        raise click.ClickException(
            "No HTTP(S) data links found with allowed extensions "
            f"{sorted(ALLOWED_EXT)}. Try relaxing filters or inspect the search JSON."
        )

    if not urls:
        raise click.ClickException("No HTTP(S) links found in the search JSON.")

    if workers:
        SETTINGS.parallel_downloads = workers

    results = download_all(
        urls,
        outdir=outdir,
        overwrite=overwrite,
        resume=resume,
        dedupe_names=dedupe_names,
    )
    ok = sum(r.status in ("downloaded", "verified", "exists") for r in results)
    click.echo(f"{ok}/{len(results)} OK → {outdir or SETTINGS.download_root}")

from urllib.parse import urlparse

ALLOWED_EXT = {".zip", ".tif", ".tiff", ".h5"}

def _first_data_link(d: dict) -> str | None:
    links = d.get("links", []) or []
    # Prefer HTTPS, allowed extensions, non-OPeNDAP
    def ok(u: str) -> bool:
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            return False
        if "/opendap" in u.lower():
            return False
        ext = Path(p.path).suffix.lower()
        return ext in ALLOWED_EXT
    best = [l["href"] for l in links if isinstance(l, dict) and isinstance(l.get("href"), str) and ok(l["href"])]
    if best:
        return best[0]
    # last resort: any http(s) link
    fallback = [l["href"] for l in links if isinstance(l, dict) and str(l.get("href","")).startswith(("http://","https://"))]
    return fallback[0] if fallback else None

# -----------------------------------------------------------------------------
# build-yaml
# -----------------------------------------------------------------------------
@main.command("build-yaml")
@click.option(
    "--search",
    "search_json",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the search results JSON produced by `dswx_enumerator search`.",
)
@click.option(
    "--template",
    type=click.Path(path_type=Path),
    required=False,
    help="Jinja2 template for the DSWx-enumerator runconfig (e.g., templates/runconfig_dswx_ni.j2).",
)
@click.option(
    "--paths",
    "paths_yaml",
    type=click.Path(path_type=Path),
    required=True,
    help="YAML of path aliases referenced by the Jinja2 template (e.g., templates/paths.yml).",
)
@click.option(
    "--yaml-out",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write rendered runconfig YAML files.",
)
@click.option(
    "--params",
    default=None,
    help=(
        "JSON dict of extra parameters made available as 'params' in the template.\n"
        "Example: --params '{\"threads\":16,\"tile_size\":4096}'"
    ),
)
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory to search for already-downloaded data files. "
        "Default: DSWXNI_DOWNLOAD_ROOT (currently: "
        f"{SETTINGS.download_root})."
    ),
)
def build_yaml_cmd(
    search_json: Path,
    template: Path,
    paths_yaml: Path,   # kept for compatibility; not used by the current template
    yaml_out: Path,
    params: Optional[str],
    input_dir: Optional[Path],
):
    """
    Render one runconfig YAML per *downloaded* granule.

    For each granule in the search JSON we:
      1) Pick the best data link (HTTPS, non-OPeNDAP, allowed ext).
      2) Compute the expected local filename.
      3) Check that file exists in --input-dir (or DSWXNI_DOWNLOAD_ROOT).
      4) If present, render the runconfig using that *actual* local path.
    """
    data = _read_json(search_json)

    try:
        params_d = json.loads(params) if params else {}
        if params and not isinstance(params_d, dict):
            raise ValueError("params JSON must be an object/dict")
    except Exception as e:
        raise click.BadParameter(f"--params must be valid JSON object: {e}")

    root = input_dir or SETTINGS.download_root
    root = Path(root)
    yaml_out.mkdir(parents=True, exist_ok=True)

    made = 0
    missing = 0
    skipped = 0
    input_files = []
    if template is None:
        template = _default_ni_template_path()
    if not template.exists():
        raise click.ClickException(f"Runconfig template not found: {template}")
    click.echo(f"Using template: {template}")
    for d in data:
        url = _first_data_link(d)
        if not url:
            skipped += 1
            continue

        # Expected local filename from URL path (ignore query)
        name = Path(urlparse(url).path).name
        if not name:
            skipped += 1
            continue

        local_path = (root / name).resolve()
        if not local_path.exists():
            click.echo(f"⤬ missing: {local_path}  (from {name})")
            missing += 1
            continue
        else:
            input_files.append(str(local_path))
        # Build YAML for this concrete file
        granule_id = d.get("title") or d.get("id") or name
    text = render_runconfig(
        template_path=Path(template),
        local_path=str(local_path),
        params=params_d,
        input_files=(input_files)  # uncomment if your template expects a list
        )

    out_file = yaml_out / f"{d.get('id', name)}.yml"
    out_file.write_text(text)
    click.echo(f"✓ {out_file.name}")
    made += 1

    click.echo(f"\nWrote {made} runconfigs → {yaml_out}")
    if missing:
        click.echo(f"Note: {missing} file(s) referenced in the search JSON were not found under {root}")
    if skipped:
        click.echo(f"Skipped {skipped} item(s) without a suitable data link")


@main.command("enumerate")
@click.option(
    "--short-name",
    help=(
        "CMR collection short name (e.g., 'ALOS_PALSAR_RTC_HiRes'). "
        "If no --concept-id is provided, the tool will try to resolve the Concept-ID."
    ),
)
@click.option(
    "--concept-id",
    help=(
        "CMR collection Concept-ID (e.g., 'C1206487504-ASF'). "
        "Preferred for speed/precision; overrides --short-name if both are given."
    ),
)
@click.option(
    "--provider",
    help=(
        "Provider/data center hint (e.g., 'ASF', 'POCLOUD'). "
        "Used to help resolve Concept-ID from --short-name."
    ),
)
@click.option(
    "--version",
    help=(
        "Collection version (e.g., '003'). Used only as a hint when resolving Concept-ID."
    ),
)
# ---- spatial: bbox/polygon OR db + mgrs_set_id ----
@click.option(
    "--bbox",
    help=(
        "Search bounding box as 'minlon,minlat,maxlon,maxlat' (WGS84). "
        "Example: --bbox '-124,45,-121.5,46.5'."
    ),
)
@click.option(
    "--polygon",
    help=(
        "Search polygon in WKT (lon/lat). Use either --bbox or --polygon (not both). "
        "Example: \"POLYGON((-122.6 37.1,-122.6 37.7,-121.9 37.7,-121.9 37.1,-122.6 37.1))\""
    ),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    help=(
        "Path to local vector database (GeoPackage/SpatiaLite/etc.) to look up polygons. "
        "Used only with --mgrs-set-id."
    ),
)
@click.option(
    "--mgrs-set-id",
    help=(
        "MGRS set identifier to look up in --db. "
        "If provided (with --db), overrides --bbox/--polygon."
    ),
)
@click.option(
    "--mgrs-layer",
    default=None,
    help=(
        "Optional layer name in the vector DB (e.g., 'polygons'). "
        "If omitted, the default layer is used."
    ),
)
@click.option(
    "--mgrs-id-col",
    default="mgrs_set_id",
    show_default=True,
    help=(
        "Column name in --db/--mgrs-layer that matches --mgrs-set-id "
        "(e.g., 'mgrs_set_id' or 'name')."
    ),
)
# ---- temporal / paging ----
@click.option(
    "--start",
    help=(
        "Start time (ISO8601). Example: 2007-01-01T00:00:00Z"
    ),
)
@click.option(
    "--end",
    help=(
        "End time (ISO8601). Example: 2011-12-31T23:59:59Z"
    ),
)
@click.option(
    "--max",
    "max_items",
    type=int,
    default=1000,
    show_default=True,
    help=(
        "Maximum number of granules to return (across pages). "
        "If using DB polygons that split into parts, this applies per part."
    ),
)
# ---- track selection policy ----
@click.option(
    "--track",
    help=(
        "Optional track number to select (e.g., '777'). "
        "REQUIRED to proceed with download/YAML if multiple tracks are present. "
        "If omitted, the command prints a summary grouped by track and exits."
    ),
)
# ---- download controls ----
@click.option(
    "--outdir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Destination directory for downloads. "
        "Default: DSWXNI_DOWNLOAD_ROOT (currently: "
        f"{SETTINGS.download_root})."
    ),
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help=(
        "Number of parallel download workers. "
        f"Default: DSWXNI_PARALLEL_DOWNLOADS (currently: {SETTINGS.parallel_downloads})."
    ),
)
@click.option(
    "--progress",
    type=click.Choice(["bytes","files","none"], case_sensitive=False),
    default="bytes",
    show_default=True,
    help=(
        "Progress display for downloads: 'bytes' shows a live bytes bar, "
        "'files' shows one tick per completed file, 'none' disables progress."
    ),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Re-download even if the destination file already exists.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Attempt to resume partial downloads when the server supports HTTP Range.",
)
@click.option(
    "--dedupe-names/--no-dedupe-names",
    default=False,
    show_default=True,
    help="Append '__N' to filenames to avoid collisions instead of overwriting.",
)
# ---- yaml rendering ----
@click.option(
    "--template",
    type=click.Path(path_type=Path),
    required=False,
    help="Jinja2 template for the DSWx-NI runconfig (e.g., templates/runconfig_dswx_ni.j2).",
)
@click.option(
    "--yaml-out",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write rendered runconfig YAML files.",
)
@click.option(
    "--params",
    default=None,
    help=(
        "JSON dict of extra parameters made available as 'params' in the template.\n"
        "Example: --params '{\"threads\":16,\"tile_size\":4096}'"
    ),
)
def enumerate_cmd(
    short_name: Optional[str],
    concept_id: Optional[str],
    provider: Optional[str],
    version: Optional[str],
    bbox: Optional[str],
    polygon: Optional[str],
    db_path: Optional[Path],
    mgrs_set_id: Optional[str],
    mgrs_layer: Optional[str],
    mgrs_id_col: str,
    start: Optional[str],
    end: Optional[str],
    max_items: int,
    track: Optional[str],
    outdir: Optional[Path],
    workers: Optional[int],
    progress: str,
    overwrite: bool,
    resume: bool,
    dedupe_names: bool,
    template: Path,
    yaml_out: Path,
    params: Optional[str],
):
    """
    End-to-end: search → (optional) track filter → download → build YAMLs.

    Behavior:
      • If --track is omitted and results include >1 track, prints a summary and exits (no downloads).
      • If --track is provided, filters to that track and proceeds with download + YAML.
    """
    # ----------------------
    # Validate identity
    # ----------------------
    if not concept_id and not short_name:
        raise click.UsageError(
            "Provide either --concept-id or --short-name (with optional --provider/--version)."
        )
    if not concept_id and short_name:
        concept_id = resolve_collection_concept_id(short_name, provider=provider, version=version)
    spec = CollectionSpec(short_name=short_name, concept_id=concept_id, provider=provider, version=version)

    # ----------------------
    # Spatial selection
    # ----------------------
    using_db = bool(db_path and mgrs_set_id)
    using_bbox_poly = bool(bbox or polygon)
    if using_db and using_bbox_poly:
        raise click.BadParameter("Choose ONE spatial mode: either --db + --mgrs-set-id OR --bbox/--polygon.")
    if not using_db and not using_bbox_poly:
        raise click.BadParameter("Provide spatial constraints: either --bbox/--polygon OR --db + --mgrs-set-id.")

    # Collect granules (merge if multiple polygon parts)
    temporal = Temporal(start=start, end=end)
    granules_list = []

    if using_db:
        cmr_polys = _polygons_from_db_general(
            db_path=db_path,
            mgrs_set_id=mgrs_set_id,         # type: ignore[arg-type]
            layer=mgrs_layer,
            id_col=mgrs_id_col,
        )
        click.echo(f"Using {len(cmr_polys)} polygon part(s) from DB for mgrs_set_id={mgrs_set_id}")

        by_id: dict[str, dict] = {}
        for i, poly_str in enumerate(cmr_polys, 1):
            spatial = Spatial(polygon_wkt=poly_str)
            part = search_cmr_flexible(
                spec,
                spatial=spatial,
                temporal=temporal,
                max_items=max_items,
            )
            for g in part:
                by_id[g.id] = g.to_dict()
            click.echo(f"  part {i}: {len(part)} granules")
        granules_list = list(by_id.values())
    else:
        if bbox and polygon:
            raise click.BadParameter("Use either --bbox or --polygon, not both.")
        spatial = Spatial(polygon_wkt=polygon) if polygon else Spatial(bbox=_parse_bbox(bbox))
        granules = search_cmr_flexible(
            spec,
            spatial=spatial,
            temporal=temporal,
            max_items=max_items,
        )
        granules_list = [g.to_dict() for g in granules]

    if not granules_list:
        click.echo("No granules found for the given constraints.")
        return

    # ----------------------
    # Track grouping / policy
    # ----------------------
    from collections import defaultdict
    by_track: dict[str, list[dict]] = defaultdict(list)
    for d in granules_list:
        t = _extract_track(d) or "unknown"
        by_track[t].append(d)

    if not track:
        # summarize and exit (no downloads)
        multi = len([k for k in by_track.keys() if k != "unknown"]) > 1
        if multi:
            click.echo("⚠️  Multiple tracks detected; this may be unexpected.\n")
        total = sum(len(v) for v in by_track.values())
        click.echo(f"Total granules: {total}")
        for k in sorted(by_track.keys(), key=lambda k: (k == "unknown", k)):
            rows = by_track[k]
            click.echo(f"- track={k} : {len(rows)}")
            for trow in rows[:3]:
                click.echo(f"    • {trow.get('title') or trow.get('id')}")
        click.echo("\nTip: re-run with --track <number> to select and proceed with download + YAML.")
        return

    # Filter to selected track
    selected = by_track.get(track) or []
    if not selected:
        click.echo(f"No granules found for track={track}. Available: {', '.join(sorted(k for k in by_track.keys() if k!='unknown')) or '(none)'}")
        return
    click.echo(f"Proceeding with track={track}: {len(selected)} granules")

    # ----------------------
    # Select data URLs (allowed extensions only)
    # ----------------------
    from urllib.parse import urlparse
    ALLOWED_EXT = {".zip", ".tif", ".tiff", ".h5"}

    def pick_url(d: dict) -> tuple[str|None, str]:
        url = _first_data_link(d)
        return url, (d.get("id") or d.get("title") or "granule")

    url_to_meta: dict[str, dict] = {}
    urls: list[str] = []
    for d in selected:
        u, _ = pick_url(d)
        if not u:
            continue
        # enforce extension filter
        ext = Path(urlparse(u).path).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        if u not in url_to_meta:
            url_to_meta[u] = d
            urls.append(u)

    if not urls:
        click.echo("No downloadable data links (zip/tif/tiff/h5) found for the selected track.")
        return

    # ----------------------
    # Download
    # ----------------------
    if workers:
        SETTINGS.parallel_downloads = workers
    dl_root = Path(outdir or SETTINGS.download_root)
    click.echo(f"Downloading {len(urls)} file(s) → {dl_root}")

    results = download_all(
        urls,
        outdir=dl_root,
        overwrite=overwrite,
        resume=resume,
        dedupe_names=dedupe_names,
        show_progress=progress,  # uses your tqdm integration
    )

    ok = [r for r in results if r.status in ("downloaded", "verified", "exists") and r.path]
    if not ok:
        click.echo("No files downloaded successfully; aborting YAML generation.")
        return
    click.echo(f"{len(ok)}/{len(results)} OK")

    # ----------------------
    # Build YAMLs only for files that exist
    # ----------------------
    if template is None:
        template = _default_ni_template_path()
    if not template.exists():
        raise click.ClickException(f"Runconfig template not found: {template}")
    click.echo(f"Using template: {template}")

    try:
        params_d = json.loads(params) if params else {}
        if params and not isinstance(params_d, dict):
            raise ValueError("params JSON must be an object/dict")
    except Exception as e:
        raise click.BadParameter(f"--params must be valid JSON object: {e}")

    yaml_out.mkdir(parents=True, exist_ok=True)
    input_files = []
    made = 0
    for r in ok:
        d = url_to_meta.get(r.url, {})
        local_path = str(Path(r.path).resolve())
        granule_id = d.get("title") or d.get("id") or Path(local_path).stem
        input_files.append(str(local_path))

    text = render_runconfig(
        template_path=Path(template),
        local_path=local_path,
        params=params_d,
        input_files=(input_files)
    )
    out_file = yaml_out / f"{d.get('id', Path(local_path).stem)}.yml"
    out_file.write_text(text)
    click.echo(f"✓ {out_file.name}")
    made += 1

    click.echo(f"\nWrote {made} runconfigs → {yaml_out}")

# -----------------------------------------------------------------------------
# resolve-cid (utility)
# -----------------------------------------------------------------------------
@main.command("resolve-cid")
@click.option("--short-name", required=True, help="Collection short name to resolve.")
@click.option("--provider", help="Provider hint (e.g., ASF, POCLOUD).")
@click.option("--version", help="Collection version (e.g., 003).")
def resolve_cid_cmd(short_name: str, provider: Optional[str], version: Optional[str]):
    """
    Resolve a collection Concept-ID (CID) for a given short name.

    Uses (in order): env override map, built-in registry, and a live CMR query.

    Example:
      dswx_enumerator resolve-cid --short-name ALOS_PALSAR_RTC_HiRes --provider ASF
    """
    cid = resolve_collection_concept_id(short_name, provider=provider, version=version)
    if not cid:
        raise click.ClickException("No Concept-ID found.")
    click.echo(cid)
