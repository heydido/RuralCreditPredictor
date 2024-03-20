from dataclasses import dataclass
from pathlib import Path


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
    schema_file: dict
