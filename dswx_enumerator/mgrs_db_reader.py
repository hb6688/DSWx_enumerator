from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import split
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient


def _geom_to_cmr_polygon_str(geom: BaseGeometry) -> str:
    """
    Convert a polygon to CMR 'polygon' string: lon1,lat1,lon2,lat2,...,lon1,lat1
    - Drops Z if present
    - Uses exterior ring only (CMR ignores holes)
    - Enforces counter-clockwise orientation (required by CMR)
    """
    if not isinstance(geom, Polygon):
        raise ValueError("Expected a Polygon geometry")

    # Ensure counter-clockwise ring
    g = orient(geom, sign=1.0)

    coords = []
    for c in g.exterior.coords:
        x, y = c[0], c[1]            # x=lon, y=lat (drop Z if present)
        if not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0):
            raise ValueError(f"Out-of-range coord for CMR polygon: lon={x}, lat={y}")
        coords.append((x, y))

    # Ensure closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return ",".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)



def _maybe_split_antimeridian(geom: BaseGeometry) -> list[BaseGeometry]:
    """
    If a polygon crosses the antimeridian, split along x=180/-180.
    Returns a list of polygonal parts in [-180, 180] longitude space.
    Safe no-op if nothing crosses.
    """
    if geom.is_empty:
        return []
    # fast path: if bbox is narrow, don't bother
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    # normalize cases like minx ~ 170, maxx ~ -170 (wrapped datasets)
    if width < 0 or width > 200:  # heuristic; 200 deg span implies wrap
        width = (maxx + 360) - minx  # coarse width under wrap assumption
    if width <= 180:
        # still may sit exactly on 180/-180; attempt split but ignore failures
        try:
            parts = split(geom, LineString([(180, -90), (180, 90)]))
            pieces = [p for p in parts.geoms] if hasattr(parts, "geoms") else [geom]
            out = []
            for p in pieces:
                more = split(p, LineString([(-180, -90), (-180, 90)]))
                out.extend([q for q in (more.geoms if hasattr(more, "geoms") else [p])])
            return out
        except Exception:
            return [geom]

    # Wide span: do both splits
    try:
        s1 = split(geom, LineString([(180, -90), (180, 90)]))
        pieces = [p for p in s1.geoms] if hasattr(s1, "geoms") else [geom]
    except Exception:
        pieces = [geom]
    out = []
    for p in pieces:
        try:
            s2 = split(p, LineString([(-180, -90), (-180, 90)]))
            out.extend([q for q in (s2.geoms if hasattr(s2, "geoms") else [p])])
        except Exception:
            out.append(p)
    return out


def _polygons_from_db_general(
    db_path: Path,
    mgrs_set_id: str,
    *,
    layer: str | None = None,        # for GeoPackage/SpatiaLite, you can name the layer (use your --mgrs-table option)
    id_col: str = "mgrs_set_id",     # matches your DB column
) -> list[str]:
    """
    Load polygons with GeoPandas, filter by mgrs_set_id, split at antimeridian,
    and return a list of WKT polygons.
    """
    if not db_path.exists():
        raise click.ClickException(f"Database not found: {db_path}")
    try:
        gdf = gpd.read_file(db_path.as_posix(), layer=layer) if layer else gpd.read_file(db_path.as_posix())
    except Exception as e:
        raise click.ClickException(f"Failed to read vector DB with GeoPandas: {e}")

    if id_col not in gdf.columns:
        raise click.ClickException(f"Column '{id_col}' not found in layer{' '+layer if layer else ''}.")

    sub = gdf[gdf[id_col] == mgrs_set_id]
    if sub.empty:
        raise click.ClickException(f"No rows found for {id_col}={mgrs_set_id}.")

    cmr_polys: list[str] = []
    for geom in sub.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = []
        if isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                parts.extend(_maybe_split_antimeridian(p))
        else:
            parts.extend(_maybe_split_antimeridian(geom))

        for p in parts:
            if isinstance(p, Polygon):
                cmr_polys.append(_geom_to_cmr_polygon_str(p))
            elif isinstance(p, MultiPolygon):
                for q in p.geoms:
                    cmr_polys.append(_geom_to_cmr_polygon_str(q))
            # ignore non-polygons

    if not cmr_polys:
        raise click.ClickException("Geometry exists but yielded no polygon parts after splitting.")
    return cmr_polys