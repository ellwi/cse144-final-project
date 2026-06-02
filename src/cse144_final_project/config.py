from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import yaml

class DataConfig(BaseModel):
    data_dir: Path = Path("./data/train")
    batch_size: int = 32
    num_workers: int = 2
    shuffle: bool = True


class ModelConfig(BaseModel):
    model_name: str
    num_classes: int = 100
    checkpoint_path: Optional[Path] = None

    unfreeze_classifier_layers: int = 1
    unfreeze_backbone_layers: int = 0


class OptimizerConfig(BaseModel):
    feature_lr: float = 1e-5
    classifier_lr: float = 1e-4
    weight_decay: float = 1e-4


class TrainingConfig(BaseModel):
    epochs: int = 10
    seed: int = 42
    label_smoothing: float = 0.1


class OutputConfig(BaseModel):
    out_dir: Path = Path("./outputs")


class TransferLearningConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> TransferLearningConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    return TransferLearningConfig(**data)