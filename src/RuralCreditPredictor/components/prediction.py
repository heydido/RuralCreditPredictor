import json
import sys
import pickle
import pandas as pd

import mlflow

from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.entity.config_entity import PredictionConfig
from src.RuralCreditPredictor.config.configuration import ConfigurationManager

import warnings
warnings.filterwarnings("ignore")


class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def load_data_transformer(self):
        try:
            logging.info("> Loading the data transformer:")

            data_transformer_path = self.config.data_transformer
            with open(data_transformer_path, 'rb') as file:
                data_transformer = pickle.load(file)

            logging.info("Data transformer loaded successfully!")

            return data_transformer

        except Exception as e:
            logging.error(f"Error in loading the data transformer: {str(e)}")
            raise CustomException("Error in loading the data transformer")

    def load_model(self):
        try:
            logging.info("> Loading the model:")

            logging.info(f"Loading run_id to track model metrics:")
            with open(self.config.latest_run_id, 'r') as file:
                run_id = file.read().strip()
                logging.info(f"run_id: {run_id}")

            experiment_id = mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id
            model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"

            model = mlflow.sklearn.load_model(model_path)

            logging.info("Model loaded successfully!")

            return model

        except Exception as e:
            logging.error(f"Error in loading the model: {str(e)}")
            raise CustomException("Error in loading the model")

    def predict(self, prediction_datapoint):
        try:
            logging.info("> Getting prediction:")

            data_transformer = self.load_data_transformer()
            model = self.load_model()

            prediction_datapoint = data_transformer.transform(prediction_datapoint)

            prediction = model.predict(prediction_datapoint)[0]

            logging.info(f"Prediction done successfully! Loan Amount: {prediction}")

            return prediction

        except Exception as e:
            logging.error(f"Error in predicting prediction!")
            raise CustomException(e,sys)


class CustomData:
    def __init__(
            self,
            age,
            sex,
            annual_income,
            monthly_expenses,
            old_dependents,
            young_dependents,
            home_ownership,
            type_of_house,
            occupants_count,
            house_area,
            loan_tenure,
            loan_installments
    ):
        self.input_data = {
            "age": age,
            "sex": sex,
            "annual_income": annual_income,
            "monthly_expenses": monthly_expenses,
            "old_dependents": old_dependents,
            "young_dependents": young_dependents,
            "home_ownership": home_ownership,
            "type_of_house": type_of_house,
            "occupants_count": occupants_count,
            "house_area": house_area,
            "loan_tenure": loan_tenure,
            "loan_installments": loan_installments
        }

    def get_data_as_df(self):
        try:
            logging.info("> Getting data for prediction:")

            data = pd.DataFrame([self.input_data])

            logging.info("Data ready for prediction!")

            return data

        except Exception as e:
            logging.error(f"Error in getting data for prediction!")
            raise CustomException(e, sys)


if __name__ == '__main__':

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

    print("Predicting loan amount for:\n", json.dumps(custom_data.input_data, indent=4))

    input_data = custom_data.get_data_as_df()

    # Get Prediction
    config_manager = ConfigurationManager()
    prediction_config = config_manager.get_prediction_config()
    predictor = Predictor(config=prediction_config)
    loan_amount = predictor.predict(input_data)

    print(f"Predicted Loan Amount: {loan_amount}")  # Predicted Loan Amount: 8500.0
