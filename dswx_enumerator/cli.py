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
import subprocess
import yaml

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
import dswx_sar.dswx_s1
import dswx_sar.dswx_ni


ALLOWED_EXT = {".zip", ".tif", ".tiff", ".h5"}

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

def _default_template_path(sensor: str) -> Path:
    """
    Pick a default template based on sensor.
      alos1       -> templates/runconfig_dswx_ni.j2
      sentinel-1  -> templates/runconfig_dswx_ni_s1.j2 (falls back to NI)
      nisar       -> templates/runconfig_dswx_ni_nisar.j2 (falls back to NI)
    """
    root = Path(__file__).resolve().parents[1] / "templates"
    mapping = {
        "alos1": root / "runconfig_dswx_ni.j2",
        "sentinel-1": root / "runconfig_dswx_s1.j2",
        "nisar": root / "runconfig_dswx_ni.j2",
    }
    p = mapping.get(sensor.lower(), root / "runconfig_dswx_ni.j2")
    # Fallbacks to NI template when a sensor-specific template is absent
    if not p.exists():
        fallback = root / "runconfig_dswx_ni.j2"
        click.echo(f"Note: default template for sensor='{sensor}' not found at {p}. "
                   f"Falling back to {fallback}.")
        return fallback
    return p

_S1_KEY_RE = re.compile(r"(T\d{3}-\d{6}-IW[1-3])", re.IGNORECASE)  # group key for S1

def _is_allowed_url(u: str) -> bool:
    try:
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            return False
        if "/opendap" in u.lower():
            return False
        return Path(p.path).suffix.lower() in ALLOWED_EXT
    except Exception:
        return False


def _search_core(
    spec: CollectionSpec,
    *,
    bbox: Optional[str],
    polygon_wkt: Optional[str],
    db_path: Optional[Path],
    mgrs_set_id: Optional[str],
    layer: Optional[str],
    id_col: str,
    start: Optional[str],
    end: Optional[str],
    max_items: int,
) -> list[dict]:
    """Return normalized granule dicts (no printing, no saving)."""
    using_db = bool(db_path and mgrs_set_id)
    if using_db:
        cmr_polys = _polygons_from_db_general(
            db_path=db_path,
            mgrs_set_id=mgrs_set_id,
            layer=layer,
            id_col=id_col,
        )
        by_id: dict[str, dict] = {}
        for poly_str in cmr_polys:
            spatial = Spatial(polygon_wkt=poly_str)
            part = search_cmr_flexible(
                spec, spatial=spatial, temporal=Temporal(start=start, end=end), max_items=max_items
            )
            for g in part:
                by_id[g.id] = g.to_dict()
        return list(by_id.values())
    else:
        spatial = Spatial(polygon_wkt=polygon_wkt) if polygon_wkt else Spatial(bbox=_parse_bbox(bbox))
        granules = search_cmr_flexible(
            spec, spatial=spatial, temporal=Temporal(start=start, end=end), max_items=max_items
        )
        return [g.to_dict() for g in granules]


def _group_by_track_payload(
    payload: list[dict], *, short_name: Optional[str], concept_id: Optional[str]
) -> dict[str, list[dict]]:
    """Group payload by track (S1 uses producer_granule_id parse)."""
    by_track: dict[str, list[dict]] = defaultdict(list)
    s1 = (short_name == "OPERA_L2_RTC-S1_V1") or (concept_id == "C2777436413-ASF")
    for d in payload:
        t = _extract_track_s1(d) if s1 else (_extract_track(d) or "unknown")
        by_track[t].append(d)
    return by_track


def _urls_from_payload(payload: list[dict]) -> list[str]:
    """All candidate links from all granules (filtered, ranked, deduped by path)."""
    urls: list[str] = []
    seen_paths: set[str] = set()
    for d in payload:
        for u in _data_links_from_granule(d):
            path = urlparse(u).path
            if path in seen_paths:
                continue
            seen_paths.add(path)
            urls.append(u)
    return urls


def _granule_files_present(
    payload: list[dict], *, root: Path
) -> dict[str, list[Path]]:
    """Map granule-id/title → list of files that actually exist under root."""
    granule_files: dict[str, list[Path]] = {}
    for d in payload:
        gid = d.get("id") or d.get("title") or "granule"
        files: list[Path] = []
        for u in _data_links_from_granule(d):
            name = Path(urlparse(u).path).name
            if not name:
                continue
            p = (root / name).resolve()
            if p.exists():
                files.append(p)
        if files:
            granule_files[gid] = sorted(files, key=lambda x: x.name)
    return granule_files

def _read_product_path(yaml_path: str):
    """
    Read product output path
    """

    yaml_config = Path(yaml_path).expanduser().resolve()

    # --- Path checks ---
    if not yaml_config.exists():
        raise FileNotFoundError(f"Runconfig YAML configuration file not found: {yaml_config}")

    if yaml_config.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {yaml_config}")

    with yaml_config.open("r") as f:
        cfg = yaml.safe_load(f)

    try:
        prod_relative_path = Path(cfg["runconfig"]["groups"]["product_path_group"]["sas_output_path"])
    except KeyError as e:
        raise KeyError("Missing runconfig.groups.product_path_group.product_path") from e

    if prod_relative_path.is_absolute():
        return prod_relative_path
    
    # Resolve relative paths against the YAML's directory
    product_abs_path = (yaml_config.parent.parent / prod_relative_path).resolve()

    return product_abs_path

def _run_dswx_sar(
    yaml_config,
    sensor,
):
    if sensor == 'sentinel-1':
        # Call DSWX-SAR via CLI
        module = "dswx_sar.dswx_s1"
    elif sensor == 'nisar':
        module = "dswx_sar.dswx_ni"

    yaml_path = str(Path(yaml_config).resolve())

    cmd = ["python", "-m", module, yaml_path]

    subprocess.run(cmd, check=True)

def _run_dwsx_hls_acc(
        dswx_data_dir: str,
        acc_output_dir: str,
        hls_acc_script: str,
        earthdata_token: str,
    ):
    """
    Run DSWX-HLS Accuracy comparison.
    """
    if not earthdata_token:
        raise click.ClickException(
            "Missing Earthdata bearer token (--earthdata_token). Please provide a valid EBT "
        )

    acc_output_dir = str(Path(acc_output_dir).resolve())

    subprocess.run([
        'python', '-m', hls_acc_script,
        '-i', dswx_data_dir,
        '-o', acc_output_dir,
        '-t', earthdata_token,
    ], check=True)


def _render_by_sensor(
    *, sensor: str, template: Path, granule_files: dict[str, list[Path]],
    yaml_out: Path, params_d: dict, organize_s1: bool = False, root: Optional[Path] = None
) -> int:
    """Render YAMLs and return count created."""
    made = 0

    sensor_key = sensor.lower()
    if sensor_key == "alos1":
        # One YAML per file
        # for p in local_files:
        files = [granule_files.items]
        text = render_runconfig(
            template_path=template,
            local_path=str(files[0]),
            params=params_d,
            input_files=(files)
        )
        # Prefer granule id where available
        # dmatch = next((d for d in data if Path(urlparse(_first_data_link(d) or "").path).name == p.name), {})
        # yname = f"{dmatch.get('id', p.stem)}.yml"
        out_file = yaml_out / f"{d.get('id', name)}.yml"
        # out_file = yaml_out / yname
        out_file.write_text(text)
        click.echo(f"✓ {out_file.name} (ALOS-1)")
        made += 1

    elif sensor_key == "sentinel-1":
        # Group across ALL files by Tddd-dddddd-IW[1-3]
        groups: dict[str, list[Path]] = {}
        for files in granule_files.values():
            for p in files:
                key = _s1_group_key_from_name(p.name)
                if key:
                    groups.setdefault(key, []).append(p)
                else:
                    click.echo(f"• skipping non-S1-pattern file: {p.name}")
        group_dirs = []
        for key, files in groups.items():
            grp_dir = root / key
            grp_dir.mkdir(parents=True, exist_ok=True)
            moved: list[Path] = []
            for src in files:
                dest = grp_dir / src.name
                if src.resolve() != dest.resolve():
                    dest.write_bytes(src.read_bytes())
                    src.unlink()
                moved.append(dest)
            groups[key] = sorted(moved, key=lambda x: x.name)
            group_dirs.append(grp_dir)
            click.echo(f"Organized {len(moved)} file(s) → {grp_dir}")

        # for key, files in sorted(groups.items()):
        text = render_runconfig(
            template_path=template,
            local_path=str(group_dirs[0]),               # representative
            input_files=group_dirs,    # ALL files in group
            params=params_d,
        )
        out_file = yaml_out / f"S1_{key}.yml"
        out_file.write_text(text)
        click.echo(f"✓ {out_file.name} (S1 group: {key}, {len(files)} files)")
        made += 1

    elif sensor_key == "nisar":
        # Treat like single-file per granule; use NISAR template if present else NI (already handled)
        text = render_runconfig(
            template_path=template,
            local_path=str(files[0]),
            params=params_d,
            input_files=(files)

        )
        dmatch = next((d for d in data if Path(urlparse(_first_data_link(d) or "").path).name == p.name), {})
        yname = f"{dmatch.get('id', p.stem)}.yml"
        out_file = yaml_out / yname
        out_file.write_text(text)
        click.echo(f"✓ {out_file.name} (NISAR)")
        made += 1

    else:
        raise click.ClickException(f"Unsupported sensor: {sensor}")

    # Summary
    click.echo(f"\nWrote {made} runconfigs → {yaml_out}")
    click.echo(f"\nWrote  → {out_file}")

    return made, out_file

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
from collections.abc import Iterable

def _as_str_list(x) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, Iterable):
        return [s for s in x if isinstance(s, str)]
    return []


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
    urls: list[str] = []
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
            urls.extend(_as_str_list(best))
    # De-duplicate while preserving order
    seen: set[str] = set()
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

def _first_data_link(d: dict) -> str | None:
    links = d.get("links", []) or []
    best = [
        l["href"] for l in links
        if isinstance(l, dict) and isinstance(l.get("href"), str) and _is_allowed_url(l["href"])
    ]
    if best:
        https_first = [u for u in best if u.startswith("https://")]
        return (https_first or best)[0]   # <- pick ONE
    fb = [
        l["href"] for l in links
        if isinstance(l, dict) and str(l.get("href","")).startswith(("http://", "https://"))
    ]
    return fb[0] if fb else None

def _s1_group_key_from_name(name: str) -> str | None:
    m = _S1_KEY_RE.search(name)
    return m.group(1) if m else None

from urllib.parse import urlparse, urlunparse
from pathlib import Path


def _normalize_url(u: str) -> str:
    """Drop query/fragment so identical files with different tokens dedupe cleanly."""
    p = urlparse(u)
    return urlunparse(p._replace(query="", fragment=""))

def _data_links_from_granule(d: dict) -> list[str]:
    """
    Return ALL candidate download URLs for a granule, filtered and ranked.
    - http/https only
    - avoid OPeNDAP
    - extension in ALLOWED_EXT
    - prefer https, data-ish rel/type, allowed suffix
    - stable-dedup by path
    """
    links = d.get("links") or []
    scored: list[tuple[int, str]] = []

    for l in links:
        if not isinstance(l, dict):
            continue
        href = l.get("href")
        if not isinstance(href, str):
            continue
        u = _normalize_url(href)
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            continue
        low = u.lower()
        if "/opendap" in low or "/dap/" in low:
            continue
        ext = Path(p.path).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue

        # score
        score = 0
        if p.scheme == "https":
            score += 4
        rel = (l.get("rel") or "").lower()
        typ = (l.get("type") or "").lower()
        if rel.endswith("data#"):
            score += 6
        if any(k in typ for k in ("geotiff", "tiff")):
            score += 3
        if any(k in typ for k in ("hdf5", "hdf", "netcdf", "zip")):
            score += 2
        if "browse" in rel or "browse" in typ:
            score -= 5

        scored.append((score, u))

    # sort & path-dedup
    scored.sort(key=lambda t: t[0], reverse=True)
    seen_paths: set[str] = set()
    out: list[str] = []
    for _, u in scored:
        path = urlparse(u).path
        if path in seen_paths:
            continue
        seen_paths.add(path)
        out.append(u)
    return out
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
    "--sensor",
    type=click.Choice(["alos1", "sentinel-1", "nisar"], case_sensitive=False),
    required=True,
    help=(
        "Sensor family for these granules: 'alos1', 'sentinel-1', or 'nisar'. "
        "Selects the default template and any sensor-specific grouping/logic."
    ),
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
    sensor: str,
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

    tmpl = template or _default_template_path(sensor)
    if not tmpl.exists():
        raise click.ClickException(f"Runconfig template not found: {tmpl}")
    click.echo(f"Using template: {tmpl}")

    root = input_dir or SETTINGS.download_root
    root = Path(root)
    yaml_out.mkdir(parents=True, exist_ok=True)


    granule_files: dict[str, list[Path]] = {}  # key = granule id/title/fallback, value = files present
    missing_files = 0
    skipped = 0

    for d in _read_json(search_json):
        urls = _data_links_from_granule(d)  # <-- MULTIPLE urls now
        if not urls:
            skipped += 1
            continue

        gid = d.get("id") or d.get("title") or "granule"
        files: list[Path] = []
        for u in urls:
            name = Path(urlparse(u).path).name
            if not name:
                continue
            p = (root / name).resolve()
            if p.exists():
                files.append(p)
            else:
                missing_files += 1

        if files:
            granule_files[gid] = sorted(files, key=lambda x: x.name)
    made = 0

    # ---------- Sensor routing ----------
    sensor_key = sensor.lower()
    if sensor_key == "alos1":
        # One YAML per file
        # for p in local_files:
        text = render_runconfig(
            template_path=tmpl,
            local_path=str(files[0]),
            params=params_d,
            input_files=(files)
        )
        # Prefer granule id where available
        # dmatch = next((d for d in data if Path(urlparse(_first_data_link(d) or "").path).name == p.name), {})
        # yname = f"{dmatch.get('id', p.stem)}.yml"
        out_file = yaml_out / f"{d.get('id', name)}.yml"
        # out_file = yaml_out / yname
        out_file.write_text(text)
        click.echo(f"✓ {out_file.name} (ALOS-1)")
        made += 1

    elif sensor_key == "sentinel-1":
        # Group across ALL files by Tddd-dddddd-IW[1-3]
        groups: dict[str, list[Path]] = {}
        for files in granule_files.values():
            for p in files:
                key = _s1_group_key_from_name(p.name)
                if key:
                    groups.setdefault(key, []).append(p)
                else:
                    click.echo(f"• skipping non-S1-pattern file: {p.name}")
        group_dirs = []
        for key, files in groups.items():
            grp_dir = root / key
            grp_dir.mkdir(parents=True, exist_ok=True)
            moved: list[Path] = []
            for src in files:
                dest = grp_dir / src.name
                if src.resolve() != dest.resolve():
                    dest.write_bytes(src.read_bytes())
                    src.unlink()
                moved.append(dest)
            groups[key] = sorted(moved, key=lambda x: x.name)
            group_dirs.append(grp_dir)
            click.echo(f"Organized {len(moved)} file(s) → {grp_dir}")

        # for key, files in sorted(groups.items()):
        text = render_runconfig(
            template_path=tmpl,
            local_path=str(group_dirs[0]),               # representative
            input_files=group_dirs,    # ALL files in group
            params=params_d,
        )
        out_file = yaml_out / f"S1_{key}.yml"
        out_file.write_text(text)
        click.echo(f"✓ {out_file.name} (S1 group: {key}, {len(files)} files)")
        made += 1

    elif sensor_key == "nisar":
        # Treat like single-file per granule; use NISAR template if present else NI (already handled)
        text = render_runconfig(
            template_path=tmpl,
            local_path=str(files[0]),
            params=params_d,
            input_files=(files)

        )
        dmatch = next((d for d in data if Path(urlparse(_first_data_link(d) or "").path).name == p.name), {})
        yname = f"{dmatch.get('id', p.stem)}.yml"
        out_file = yaml_out / yname
        out_file.write_text(text)
        click.echo(f"✓ {out_file.name} (NISAR)")
        made += 1

    else:
        raise click.ClickException(f"Unsupported sensor: {sensor}")

    # Summary
    click.echo(f"\nWrote {made} runconfigs → {yaml_out}")
    if missing_files:
        click.echo(f"Note: {missing_files} file(s) referenced in the search JSON were not found under {root}")
    if skipped:
        click.echo(f"Skipped {skipped} item(s) without a suitable data link")


# -----------------------------------------------------------------------------
# enumerate
# -----------------------------------------------------------------------------
@main.command("enumerate")
@click.option(
    "--short-name",
    help="Collection short name (e.g., 'ALOS_PALSAR_RTC_HiRes', 'OPERA_L2_RTC-S1_V1')."
)
@click.option(
    "--concept-id",
    help="Collection Concept-ID (e.g., 'C1206487504-ASF'). Overrides --short-name if both are given."
)
@click.option("--provider", help="Provider hint (e.g., 'ASF', 'POCLOUD').")
@click.option("--version", help="Collection version (e.g., '003').")
# spatial (one of: bbox/polygon OR db+mgrs)
@click.option(
    "--bbox",
    help="Search bounding box as 'minlon,minlat,maxlon,maxlat' (WGS84), e.g. --bbox '-124,45,-121.5,46.5'."
)
@click.option(
    "--polygon",
    help="Search polygon in WKT (lon/lat). Use either --bbox or --polygon (not both)."
)
@click.option("--db", "db_path", type=click.Path(path_type=Path), help="Vector DB/GPKG/SpatiaLite path for MGRS lookup.")
@click.option("--mgrs-set-id", help="MGRS set identifier to look up in --db.")
@click.option("--mgrs-layer", default=None, help="Optional layer/table name in the DB.")
@click.option("--mgrs-id-col", default="mgrs_set_id", show_default=True, help="ID column name in the DB layer.")
# temporal / paging
@click.option("--start", help="Start time (ISO8601).")
@click.option("--end", help="End time (ISO8601).")
@click.option("--max", "max_items", type=int, default=1000, show_default=True,
              help="Maximum number of granules to return (across pages).")
# track policy
@click.option(
    "--track",
    help=("Optional track number (e.g., '777'). "
          "If omitted and multiple tracks are present, a summary is printed and the command exits. "
          "If omitted and exactly one track is present, it is auto-selected.")
)
# download
@click.option("--outdir", type=click.Path(path_type=Path), default=None,
              help=f"Download directory. Default: DSWX_DOWNLOAD_ROOT ({SETTINGS.download_root}).")
@click.option("--workers", type=int, default=None,
              help=f"Parallel download workers. Default: DSWX_PARALLEL_DOWNLOADS ({SETTINGS.parallel_downloads}).")
@click.option("--progress", type=click.Choice(["bytes", "files", "none"], case_sensitive=False),
              default="bytes", show_default=True,
              help="Progress display for downloads.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--dedupe-names/--no-dedupe-names", default=False, show_default=True)
# YAML
@click.option(
    "--sensor",
    type=click.Choice(["alos1", "sentinel-1", "nisar"], case_sensitive=False),
    required=True,
    help="Selects the default template and minor policy for YAML rendering."
)
@click.option("--template", type=click.Path(path_type=Path), required=False,
              help="Override Jinja2 template. Defaults by --sensor.")
@click.option("--yaml-out", type=click.Path(path_type=Path), required=True,
              help="Directory to write the single rendered runconfig YAML.")
@click.option("--params", default=None,
              help="JSON dict of extra parameters exposed as 'params' in the template.")
@click.option(
    "--acc_output_dir",
    type=click.Path(path_type=Path), 
    default=None,
    help="Data directory for DSWx-SAR products."
)
@click.option(
    "--hls_acc_script",
    type=str, 
    default=None,
    help="Script which compares DSWx-SAR output against DSWx-HLS for accuracy"
)
@click.option(
    "--earthdata_token",
    type=str, 
    default=None,
    help="Earthdata login bearer token (EBT) for accessing NASA CMR/POCLOUD data."
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
    sensor: str,
    template: Optional[Path],
    yaml_out: Path,
    params: Optional[str],
    acc_output_dir: str,
    hls_acc_script: str,
    earthdata_token: str,
):
    """
    End-to-end: search → (optional) track filter → download → build ONE YAML including all files.
    """

    # --- Identity / Concept-ID ---
    if not concept_id and not short_name:
        raise click.UsageError("Provide either --concept-id or --short-name.")
    if not concept_id and short_name:
        concept_id = resolve_collection_concept_id(short_name, provider=provider, version=version)
    spec = CollectionSpec(short_name=short_name, concept_id=concept_id, provider=provider, version=version)

    # --- Spatial validation ---
    using_db = bool(db_path and mgrs_set_id)
    using_bbox_poly = bool(bbox or polygon)
    if using_db and using_bbox_poly:
        raise click.BadParameter("Choose ONE spatial mode: either --db + --mgrs-set-id OR --bbox/--polygon.")
    if not (using_db or using_bbox_poly):
        raise click.BadParameter("Provide spatial constraints: --bbox/--polygon OR --db + --mgrs-set-id.")

    # --- Search (same logic as `search`) ---
    temporal = Temporal(start=start, end=end)
    if using_db:
        cmr_polys = _polygons_from_db_general(
            db_path=db_path,
            mgrs_set_id=mgrs_set_id,
            layer=mgrs_layer,
            id_col=mgrs_id_col,
        )
        by_id: dict[str, dict] = {}
        for poly in cmr_polys:
            spatial = Spatial(polygon_wkt=poly)
            part = search_cmr_flexible(spec, spatial=spatial, temporal=temporal, max_items=max_items)
            for g in part: by_id[g.id] = g.to_dict()
        payload = list(by_id.values())
    else:
        spatial = Spatial(polygon_wkt=polygon) if polygon else Spatial(bbox=_parse_bbox(bbox))
        granules = search_cmr_flexible(spec, spatial=spatial, temporal=temporal, max_items=max_items)
        payload = [g.to_dict() for g in granules]

    if not payload:
        click.echo("No granules found for the given constraints.")
        return

    # --- Track grouping (same policy as `search`) ---
    by_track = defaultdict(list)
    is_s1 = (short_name == "OPERA_L2_RTC-S1_V1") or (concept_id == "C2777436413-ASF")
    for d in payload:
        t = _extract_track_s1(d) if is_s1 else (_extract_track(d) or "unknown")
        by_track[t].append(d)

    if not track:
        track_keys = [k for k in by_track if k != "unknown"]
        if len(track_keys) > 1:
            # print summary and exit
            click.echo("⚠️  Multiple tracks detected; this may be unexpected.\n")
            total = sum(len(v) for v in by_track.values())
            click.echo(f"Total granules: {total}")
            for k in sorted(by_track.keys(), key=lambda k: (k == "unknown", k)):
                rows = by_track[k]
                click.echo(f"- track={k} : {len(rows)}")
                for trow in rows[:3]:
                    click.echo(f"    • {trow.get('title') or trow.get('id')}")
            click.echo("\nTip: re-run with --track <number> to proceed with download + YAML.")
            return
        elif len(track_keys) == 1:
            track = track_keys[0]
            click.echo(f"No --track specified; auto-selecting track={track}.")
        else:
            click.echo("No track metadata detected; proceeding without track filter.")
            # leave track as None → select whole payload

    selected = by_track.get(track, payload) if track else payload
    if not selected:
        click.echo(f"No granules found for track={track}.")
        return

    # --- Collect ALL URLs from selected granules (multi-link per granule) ---
    urls: list[str] = []
    seen_paths: set[str] = set()
    for d in selected:
        for u in _data_links_from_granule(d):
            path = urlparse(u).path
            if path in seen_paths:
                continue
            seen_paths.add(path)
            urls.append(u)
    if not urls:
        click.echo("No downloadable data links (zip/tif/tiff/h5) in selected granules.")
        return

    # --- Download ---
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
        show_progress=progress,
    )
    ok = [r for r in results if r.status in ("downloaded", "verified", "exists") and r.path]
    click.echo(f"{len(ok)}/{len(results)} OK")
    if not ok:
        click.echo("No files available; aborting YAML generation.")
        return

    # --- Build ONE YAML with all successfully downloaded files ---
    try:
        params_d = json.loads(params) if params else {}
        if params and not isinstance(params_d, dict):
            raise ValueError("params JSON must be a dict/object")
    except Exception as e:
        raise click.BadParameter(f"--params must be valid JSON object: {e}")

    tmpl = Path(template) if template else _default_template_path(sensor)
    if not tmpl.exists():
        raise click.ClickException(f"Runconfig template not found: {tmpl}")

    yaml_out.mkdir(parents=True, exist_ok=True)
    granule_files = _granule_files_present(selected, root=dl_root)

    made, yaml_config = _render_by_sensor(
        sensor=sensor, template=tmpl, granule_files=granule_files,
        yaml_out=yaml_out, params_d=params_d,
        organize_s1=False, root=dl_root,   # set True to auto-organize S1 into group dirs
    )
    click.echo(f"\nWrote {made} runconfigs → {yaml_out}")

    click.echo(f"\n---Start running DSWx-SAR Algorithm---\n")

    # --- Run DSWx-SAR Algorithms --- 
    _run_dswx_sar(
        yaml_config, 
        sensor,
    )
    click.echo(f"\n---Completed running DSWx-SAR Algorithm---\n")

    click.echo(f"\n---Start running DSWx-HLS Accuracy Comparison---\n")

    # Run DSWx-HLS comparison for accuracy
    # Read YAML file product path
    dswx_data_dir = _read_product_path(yaml_config)

    _run_dwsx_hls_acc(
        dswx_data_dir,
        acc_output_dir,
        hls_acc_script,
        earthdata_token,
    )

    click.echo(f"\n---Completed running DSWx-HLS Accuracy Comparison---\n")

    # _render_by_sensor(
    #     *, sensor: str, template: Path, granule_files: dict[str, list[Path]],
    #     yaml_out: Path, params_d: dict, organize_s1: bool = False, root: Optional[Path] = None
    # local_files = sorted({Path(r.path).resolve() for r in ok}, key=lambda p: p.name)
    # text = render_runconfig(
    #     template_path=tmpl,
    #     local_path=str(local_files[0]),                  # representative
    #     input_files=[str(p) for p in local_files],       # ALL files in one config
    #     params=params_d,
    # )

    # # Name the YAML meaningfully
    # name_bits = [sensor.lower()]
    # if track: name_bits.append(f"T{track}")
    # name_bits.append(f"batch_{len(local_files)}")
    # out_file = yaml_out / ("_".join(name_bits) + ".yml")

    # out_file.write_text(text)
    # click.echo(f"✓ wrote {out_file.name} with {len(local_files)} input file(s)")
    # click.echo(f"\nWrote 1 runconfig → {yaml_out}")


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


# -----------------------------------------------------------------------------
# run-dswx-sar
# -----------------------------------------------------------------------------
@main.command("run_dswx_sar")
@click.option(
    "--yaml_config",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    help="YAML configuration file to run for DSWx-SAR."
)
@click.option(
    "--sensor",
    type=click.Choice(["alos1", "sentinel-1", "nisar"], case_sensitive=False),
    required=True,
    help=(
        "Sensor family for these granules: 'alos1', 'sentinel-1', or 'nisar'. "
        "Selects the default template and any sensor-specific grouping/logic."
    ),
)

def run_dswx_sar_cmd(
    yaml_config: str, 
    sensor: str,
):
    """
    Run DSWX-SAR for Sentinel-1 using a single runconfig YAML.
    """

    if sensor == 'sentinel-1':
        # Call DSWX-SAR via CLI
        module = "dswx_sar.dswx_s1"
    elif sensor == 'nisar':
        module = "dswx_sar.dswx_ni"

    yaml_path = str(Path(yaml_config).resolve())

    cmd = ["python", "-m", module, yaml_path]

    subprocess.run(cmd, check=True)

# -----------------------------------------------------------------------------
# compute-dswx-HLS_acc
# -----------------------------------------------------------------------------
@main.command("compute_dswx_hls_acc")
@click.option(
    "--dswx_data_dir",
    type=click.Path(path_type=Path), 
    default=None,
    help="Data directory for DSWx-SAR products."
)
@click.option(
    "--acc_output_dir",
    type=click.Path(path_type=Path), 
    default=None,
    help="Data directory for DSWx-SAR products."
)
@click.option(
    "--hls_acc_script",
    type=str, 
    default=None,
    help="Script which compares DSWx-SAR output against DSWx-HLS for accuracy"
)
@click.option(
    "--earthdata_token",
    type=str, 
    default=None,
    help="Earthdata login bearer token (EBT) for accessing NASA CMR/POCLOUD data."
)

def compute_dswx_hls_acc_cmd(
    dswx_data_dir: str, 
    acc_output_dir: str,
    hls_acc_script: str,
    earthdata_token: str,

):
    """
    Run DSWX-HLS Accuracy comparison.
    """

    if not earthdata_token:
        raise click.ClickException(
            "Missing Earthdata bearer token (--earthdata_token). Please provide a valid EBT "
        )

    dswx_data_dir = str(Path(dswx_data_dir).resolve())
    acc_output_dir = str(Path(acc_output_dir).resolve())

    subprocess.run([
        'python', '-m', hls_acc_script,
        '-i', dswx_data_dir,
        '-o', acc_output_dir,
        '-t', earthdata_token,
    ], check=True)   
