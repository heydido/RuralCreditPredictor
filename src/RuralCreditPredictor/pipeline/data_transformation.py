import sys
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.config.configuration import ConfigurationManager
from src.RuralCreditPredictor.components.data_transformation import DataTransformation


STAGE_NAME = "Data Transformation"


class DataTransformationPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformer_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.split_data()
        data_transformation.get_data_transformer()
        data_transformation.transform_data()
        data_transformation.save_data_transformer()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

        data_transformer = DataTransformationPipeline()
        data_transformer.main()

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logging.error(f"Error occurred while running {STAGE_NAME}!")
        raise CustomException(e, sys)