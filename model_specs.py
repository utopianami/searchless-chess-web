"""Model catalog, safe to import without initializing the GPU runtime."""
from dataclasses import dataclass

DEFAULT_MODEL = '270M'


@dataclass(frozen=True)
class ModelSpec:
    label: str
    policy: str
    layers: int
    width: int
    checkpoint: str
    estimated_vram_mib: int


MODEL_SPECS = {
    '9M': ModelSpec('9M · 빠른 분석', 'action_value', 8, 256, '9M/6400000/params', 1024),
    '136M': ModelSpec('136M · 대형 모델', 'action_value', 8, 1024, '136M/6400000/params', 2048),
    '270M': ModelSpec('270M · 최대 모델', 'action_value', 16, 1024, '270M/6400000/params', 3072),
    '9M_state_value': ModelSpec('9M · State Value', 'state_value', 8, 256, '9M_state_value', 1024),
    '9M_behavioral_cloning': ModelSpec('9M · Behavioral Cloning', 'behavioral_cloning', 8, 256, '9M_behavioral_cloning', 1024),
}
