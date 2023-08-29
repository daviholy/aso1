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

class _MergeConfig(BaseModel):
    max_seconds: float
    overlap: float

class AnomalyConfig(BaseModel):
    ICP: _TypeConfig
    ABP: _TypeConfig
    merge: _MergeConfig

def load_config() -> AnomalyConfig:
    with open(Path("./config.yaml"), "r") as file:
        return AnomalyConfig.model_validate(yaml.safe_load(file)) 
