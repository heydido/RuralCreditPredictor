import sys
from src.RuralCreditPredictor.constants import *
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.entity.config_entity import (DataIngestionConfig,
                                                           DataValidationConfig,
                                                           DataPreprocessingConfig,
                                                           DataTransformationConfig)
from src.RuralCreditPredictor.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 raw_schema_filepath=RAW_SCHEMA_FILE_PATH,
                 processed_schema_filepath=PROCESSED_SCHEMA_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.raw_schema = read_yaml(raw_schema_filepath)
        self.processed_schema = read_yaml(processed_schema_filepath)
        self.params = read_yaml(params_filepath)

        # Artifact root directory
        create_directories([self.config.root.artifact])

    def get_data_ingestion_config(self, log=True) -> DataIngestionConfig:
        try:
            if log:
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

            if log:
                logging.info("Data ingestion configuration loaded successfully!")

            return data_ingestion_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data ingestion configuration!")
            raise CustomException(e, sys)

    def get_data_validation_config(self, log=True) -> DataValidationConfig:
        try:
            if log:
                logging.info("Getting data validation configuration:")

            config = self.config.data_validation
            raw_schema = self.raw_schema.independent_variables

            create_directories([config.root_dir])

            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                raw_file=config.raw_file,
                status_file=config.status_file,
                raw_schema=raw_schema
            )

            if log:
                logging.info("Data validation configuration loaded successfully!")

            return data_validation_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data validation configuration!")
            raise CustomException(e, sys)

    def get_data_preprocessing_config(self, log=True) -> DataPreprocessingConfig:
        try:
            if log:
                logging.info("Getting data preprocessing configuration:")

            config = self.config.data_preprocessing
            cat_features = self.processed_schema.cat_features
            discrete_num_features = self.processed_schema.discrete_num_features
            continuous_num_features = self.processed_schema.continuous_num_features
            selected_features = self.processed_schema.selected_features
            target_variable = self.processed_schema.target_variable

            create_directories([config.root_dir])

            data_preprocessing_config = DataPreprocessingConfig(
                root_dir=config.root_dir,
                raw_file=config.raw_file,
                preprocessed_file=config.preprocessed_file,
                cat_features=cat_features,
                discrete_num_features=discrete_num_features,
                continuous_num_features=continuous_num_features,
                selected_features=selected_features,
                target_variable=target_variable
            )

            if log:
                logging.info("Data preprocessing configuration loaded successfully!")

            return data_preprocessing_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data preprocessing configuration!")
            raise CustomException(e, sys)

    def get_data_transformer_config(self, log=True) -> DataTransformationConfig:
        try:
            if log:
                logging.info("Getting data transformation configuration:")

            config = self.config.data_transformation
            cat_features = self.processed_schema.cat_features
            discrete_num_features = self.processed_schema.discrete_num_features
            continuous_num_features = self.processed_schema.continuous_num_features
            selected_features = self.processed_schema.selected_features
            target_variable = self.processed_schema.target_variable

            create_directories([config.root_dir])

            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                preprocessed_file=config.preprocessed_file,
                cat_features=cat_features,
                discrete_num_features=discrete_num_features,
                continuous_num_features=continuous_num_features,
                selected_features=selected_features,
                target_variable=target_variable,
                data_transformer=config.data_transformer
            )

            if log:
                logging.info("Data transformation configuration loaded successfully!")

            return data_transformation_config

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data transformation configuration!")
            raise CustomException(e, sys)
