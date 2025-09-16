from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    download_root: Path = Path.home() / "data" / "dswxni"
    timeout_s: int = 45
    parallel_downloads: int = 6
    earthdata_username: str | None = None
    earthdata_password: str | None = None
    model_config = {"env_prefix": "DSWXNI_", "env_file": ".env", "extra": "ignore"}

SETTINGS = Settings()
