import sys
from src.RuralCreditPredictor.constants import *
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.RuralCreditPredictor.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.schema = read_yaml(schema_filepath)
        self.params = read_yaml(params_filepath)

        # Artifact root directory
        create_directories([self.config.root.artifact])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            logging.info("Getting data ingestion configuration:")

            config = self.config.data_ingestion

            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                data_file=config.data_file,
                unzip_dir=config.unzip_dir,
                raw_file=config.raw_file
            )

            logging.info("Data ingestion configuration loaded successfully!")

            return data_ingestion_config

        except Exception as e:
            logging.error(f"Error occurred while getting data ingestion configuration!")
            raise CustomException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            logging.info("Getting data validation configuration")

            config = self.config.data_validation
            schema = self.schema.independent_variables

            create_directories([config.root_dir])

            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                raw_file=config.raw_file,
                status_file=config.status_file,
                schema_file=schema
            )

            logging.info("Data validation configuration loaded successfully!")

            return data_validation_config

        except Exception as e:
            logging.error(f"Error occurred while getting data validation configuration!")
            raise CustomException(e, sys)
