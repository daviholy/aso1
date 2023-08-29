from pathlib import Path

import yaml
from pydantic import BaseModel


class _AnomalySetting(BaseModel):
    th: float
    stride: float
    window_size: float

class AnomalyConfig(BaseModel):
    max: _AnomalySetting
    min: _AnomalySetting
    line:_AnomalySetting
    peaks:_AnomalySetting

def load_config() -> AnomalyConfig:
    with open(Path("./config.yaml"), 'r') as file:
        return AnomalyConfig.model_validate(yaml.safe_load(file))