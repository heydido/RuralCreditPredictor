import sys
import json
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.config.configuration import ConfigurationManager
from src.RuralCreditPredictor.components.prediction import CustomData, Predictor


STAGE_NAME = "Prediction"


class PredictionPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():

        # Data for prediction
        custom_data = CustomData(
            age=25,
            sex='M',
            annual_income=500000.0,
            monthly_expenses=10000.0,
            old_dependents=2,
            young_dependents=0,
            home_ownership=0.0,
            type_of_house='T1',
            occupants_count=4,
            house_area=800,
            loan_tenure=12,
            loan_installments=6)

        logging.info(f"Input Data: \n {json.dumps(custom_data.input_data, indent=4)}")

        input_data = custom_data.get_data_as_df()

        # Get Prediction
        config_manager = ConfigurationManager()
        prediction_config = config_manager.get_prediction_config()
        predictor = Predictor(config=prediction_config)

        return predictor.predict(input_data)


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage '{STAGE_NAME}' started <<<<<<")

        prediction = PredictionPipeline()
        loan_amount = prediction.main()

        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logging.error(f"Error occurred while running {STAGE_NAME}!")
        raise CustomException(e, sys)
