import sys
import pandas as pd
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.config.configuration import ConfigurationManager
from src.RuralCreditPredictor.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_schema(self):
        try:
            logging.info("Validating schema of raw data file:")

            validation_status = None

            data = pd.read_csv(self.config.raw_file)
            all_cols = list(data.columns)

            raw_schema = self.config.raw_schema.keys()

            for col in all_cols:
                if col not in raw_schema:
                    logging.error(f"Column '{col}' not found in schema!")

                    validation_status = False
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

                else:
                    validation_status = True
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            logging.info(f"Final validation status: {validation_status}")

            return validation_status

        except Exception as e:
            logging.error(f"Error occurred while validating schema of raw data file!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    data_validation_config = config_manager.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_schema()
