from pathlib import Path

import yaml
from pydantic import BaseModel


class _AnomalySetting(BaseModel):
    th: float
    stride: float
    window_size: float


class _TypeConfig(BaseModel):
    max: _AnomalySetting
    min: _AnomalySetting
    line: _AnomalySetting
    peaks: _AnomalySetting

class AnomalyConfig(BaseModel):
    ICP: _TypeConfig
    ABP: _TypeConfig
    distance: float

_CONFIG: None | AnomalyConfig = None
def load_config() -> AnomalyConfig:
    global _CONFIG
    
    if not _CONFIG:
        with open(Path("./config.yaml"), "r") as file:
            _CONFIG = AnomalyConfig.model_validate(yaml.safe_load(file))
    return _CONFIG
