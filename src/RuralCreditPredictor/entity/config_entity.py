from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    data_file: Path
    unzip_dir: Path
    raw_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    raw_file: Path
    status_file: Path
    raw_schema: dict


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    raw_file: Path
    preprocessed_file: Path
    cat_features: list
    discrete_num_features: list
    continuous_num_features: list
    selected_features: list
    target_variable: str


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    preprocessed_file: Path
    cat_features: list
    discrete_num_features: list
    continuous_num_features: list
    selected_features: list
    target_variable: str
    data_transformer: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model_params: dict
    model: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model: Path
    train_metrics: Path
    test_metrics: Path


@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    data_transformer: Path
    model: Path
