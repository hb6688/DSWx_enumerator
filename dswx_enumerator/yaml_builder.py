from __future__ import annotations
from pathlib import Path
from typing import Sequence
import os
import json
import re

from jinja2 import Environment, FileSystemLoader
import yaml

# ---------- Default path fallbacks (used when neither --params nor env overrides are set)
DEFAULTS = {
    # dynamic_ancillary_file_group
    "dem_file_default": "/home/shiroma/dat/nisar-dem-copernicus/EPSG4326/EPSG4326.vrt",
    "worldcover_file_default": "/mnt/aurora-r0/jungkyo/data/landcover.vrt",
    "glad_classification_file_default": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/landcover_test/glad_landcover_2020/glad_map.vrt",
    "reference_water_file_default": "/mnt/aurora-r0/jungkyo/data/pekel.vrt",
    "hand_file_default": "/mnt/aurora-r0/jungkyo/data/hand/data/EPSG4326.vrt",
    "eth_global_canopy_file_default": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/ETH_Global_canopy_height/ETH.vrt",
    "algorithm_parameters_ni_default": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/scale/algorithm_parameter_ni2.yaml",
    "algorithm_parameters_s1_default": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-final-patch/shared/input_dir/ancillary_data/algorithm_parameters_s1.yaml",

    # static_ancillary_file_group
    "mgrs_database_file_default": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/R1_interface/sample_data/input_dir/ancillary_data/MGRS_tile.sqlite",
    "mgrs_collection_database_ni_file_default": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-NI/R1_interface/sample_data/input_dir/ancillary_data/MGRS_collection_db_DSWx-NI_v0.1.sqlite",
    "mgrs_collection_database_s1_file_default": "/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-final-patch/shared/input_dir/ancillary_data/MGRS_tile_collection_v0.3.sqlite",

    # product/scratch roots (can be set by env; see `paths_from_env`)
    "product_root_default": "products",
    "scratch_root_default": "scratch",
}

ENV_TO_PATH_KEY = {
    "DSWXNI_PRODUCT_ROOT": "product_root",
    "DSWXNI_SCRATCH_ROOT": "scratch_root",
    # dynamic ancillaries
    "DSWXNI_DEM_FILE": "dem_file",
    "DSWXNI_WORLDCOVER_FILE": "worldcover_file",
    "DSWXNI_GLAD_FILE": "glad_classification_file",
    "DSWXNI_REF_WATER_FILE": "reference_water_file",
    "DSWXNI_HAND_FILE": "hand_file",
    "DSWXNI_CANOPY_FILE": "eth_global_canopy_file",
    "DSWXNI_ALGO_PARAMS": "algorithm_parameters",
    # static ancillaries
    "DSWXNI_MGRS_DB": "mgrs_database_file",
    "DSWXNI_MGRS_COLLECTION_DB": "mgrs_collection_database_file",
}

def _sanitize_basename(p: Path) -> str:
    """
    Turn an input filename into a safe product basename (no extension, no spaces).
    """
    stem = p.name
    # drop extension(s)
    stem = re.sub(r"\.(h5|hdf5|tif|tiff|nc|zip|tar\.gz|tgz)$", "", stem, flags=re.IGNORECASE)
    # normalize spaces
    stem = re.sub(r"\s+", "_", stem)
    return stem

def paths_from_env() -> dict:
    """
    Build the 'paths' dict that the template expects, merging env overrides with DEFAULTS.
    """
    paths = {
        # product/scratch roots
        "product_root": os.environ.get("DSWXNI_PRODUCT_ROOT", DEFAULTS["product_root_default"]),
        "scratch_root": os.environ.get("DSWXNI_SCRATCH_ROOT", DEFAULTS["scratch_root_default"]),
        # per-file defaults for ancillaries
        **{k: v for k, v in DEFAULTS.items() if k.endswith("_default")},
    }
    return paths

def render_runconfig(
    template_path: Path,
    *,
    local_path: str,
    input_files: Sequence[str] | None = None,
    params: dict | None = None,
) -> str:
    """
    Render the DSWx-NI runconfig from the Jinja2 template.

    Parameters
    ----------
    template_path : Path
        Path to templates/runconfig_dswx_ni.j2
    local_path : str
        Primary local input file path (RTC/GCOV).
    input_files : optional list[str]
        If provided, these will populate input_file_path (otherwise we use local_path).
    params : dict
        Arbitrary overrides exposed as 'params' inside the template.

    Returns
    -------
    str (YAML)
    """
    template_dir = template_path.parent
    env = Environment(loader=FileSystemLoader(str(template_dir)))

    template = env.get_template(template_path.name)

    input_p = Path(local_path)
    product_basename = _sanitize_basename(input_p)

    paths = paths_from_env()

    text = template.render(
        local_path=local_path,
        input_files=input_files if input_files else None,
        product_basename=product_basename,
        paths=paths,
        params=params or {},
        env=os.environ,  # allow {{ env.* }} lookups in template
    )
    return text
