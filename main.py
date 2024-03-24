import sys
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.pipeline.data_ingestion import DataIngestionPipeline
from src.RuralCreditPredictor.pipeline.data_vaildation import DataValidationPipeline
from src.RuralCreditPredictor.pipeline.data_preprocessing import DataPreprocessingPipeline
from src.RuralCreditPredictor.pipeline.data_transformation import DataTransformationPipeline
from src.RuralCreditPredictor.pipeline.model_training import ModelTrainingPipeline
from src.RuralCreditPredictor.pipeline.model_evaluation import ModelEvaluationPipeline
from src.RuralCreditPredictor.pipeline.predict import PredictionPipeline


logging.info(">>>>>> Rural Credit Predictor Pipeline started <<<<<<\n")

STAGE_NAME = "Data Ingestion"

try:
    logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

    data_ingestor = DataIngestionPipeline()
    data_ingestor.main()

    logging.info(f">>>>>> stage '{STAGE_NAME}' completed <<<<<<\n")

except Exception as e:
    logging.error(f"Error occurred while running {STAGE_NAME}!")
    raise CustomException(e, sys)


STAGE_NAME = "Data Validation"

try:
    logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

    data_validator = DataValidationPipeline()
    data_validator.main()

    logging.info(f">>>>>> stage '{STAGE_NAME}' completed <<<<<<\n")

except Exception as e:
    logging.error(f"Error occurred while running {STAGE_NAME}!")
    raise CustomException(e, sys)


STAGE_NAME = "Data Preprocessing"

try:
    logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

    data_preprocessor = DataPreprocessingPipeline()
    data_preprocessor.main()

    logging.info(f">>>>>> stage '{STAGE_NAME}' completed <<<<<<\n")

except Exception as e:
    logging.error(f"Error occurred while running {STAGE_NAME}!")
    raise CustomException(e, sys)


STAGE_NAME = "Data Transformation"

try:
    logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

    data_transformer = DataTransformationPipeline()
    data_transformer.main()

    logging.info(f">>>>>> stage '{STAGE_NAME}' completed <<<<<<\n")

except Exception as e:
    logging.error(f"Error occurred while running {STAGE_NAME}!")
    raise CustomException(e, sys)


STAGE_NAME = "Model Training"

try:
    logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

    model_trainer = ModelTrainingPipeline()
    model_trainer.main()

    logging.info(f">>>>>> stage '{STAGE_NAME}' completed <<<<<<\n")

except Exception as e:
    logging.error(f"Error occurred while running {STAGE_NAME}!")
    raise CustomException(e, sys)


STAGE_NAME = "Model Evaluation"

try:
    logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

    model_evaluator = ModelEvaluationPipeline()
    model_evaluator.main()

    logging.info(f">>>>>> stage '{STAGE_NAME}' completed <<<<<<\n")

except Exception as e:
    logging.error(f"Error occurred while running {STAGE_NAME}!")
    raise CustomException(e, sys)


STAGE_NAME = "Prediction"

try:
    logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

    predictor = PredictionPipeline()
    predictor.main()

    logging.info(f">>>>>> stage '{STAGE_NAME}' completed <<<<<<\n")
    logging.info(">>>>>> Rural Credit Predictor Pipeline completed <<<<<<")

except Exception as e:
    logging.error(f"Error occurred while running {STAGE_NAME}!")
    raise CustomException(e, sys)
