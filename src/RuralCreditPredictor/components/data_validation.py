import sys
import pandas as pd
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
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

            schema = self.config.schema_file.keys()

            for col in all_cols:
                if col not in schema:
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
