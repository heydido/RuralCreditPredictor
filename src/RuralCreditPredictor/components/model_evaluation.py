import sys
import json

import mlflow
from urllib.parse import urlparse

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.config.configuration import ConfigurationManager
from src.RuralCreditPredictor.components.model_trainer import ModelTrainer


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def evaluate_model(true, predicted, log=True):
        try:
            if log:
                logging.info("> Evaluating model:")

            mae = mean_absolute_error(true, predicted)
            mape = mean_absolute_percentage_error(true, predicted)
            mse = mean_squared_error(true, predicted)
            rmse = np.sqrt(mean_squared_error(true, predicted))

            if log:
                logging.info("Model evaluated successfully!")

            return mae, mape, mse, rmse

        except Exception as e:
            if log:
                logging.error(f"Error occurred while evaluating model!")
            raise CustomException(e, sys)

    def get_model_metrics(self):
        try:
            logging.info("> Getting model metrics:")

            logging.info(f"Loading run_id to track model metrics:")
            with open(self.config.latest_run_id, 'r') as file:
                run_id = file.read().strip()
                logging.info(f"run_id: {run_id}")

            with mlflow.start_run(run_id=run_id):
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                if tracking_url_type_store != "file":
                    model = mlflow.sklearn.load_model("models:/RandomForestModel/None")  # None to pull the latest model

                else:
                    experiment_id = mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id
                    model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
                    model = mlflow.sklearn.load_model(model_path)

                config = ConfigurationManager()
                model_training_config = config.get_model_training_config(log=False)
                trainer = ModelTrainer(config=model_training_config)

                x_train, x_test, y_train, y_test = trainer.get_data(log=False)

                y_pred_train = model.predict(x_train)
                y_pred_test = model.predict(x_test)

                mae_train, mape_train, mse_train, rmse_train = self.evaluate_model(y_train, y_pred_train, log=False)

                mlflow.log_metric("mae_train", mae_train)
                mlflow.log_metric("mape_train", mape_train)
                mlflow.log_metric("mse_train", mse_train)
                mlflow.log_metric("rmse_train", rmse_train)

                logging.info(
                    f"Train metrics: MAE: {mae_train}, MAPE: {mape_train}, MSE: {mse_train}, RMSE: {rmse_train}"
                )

                mae_test, mape_test, mse_test, rmse_test = self.evaluate_model(y_test, y_pred_test, log=False)

                mlflow.log_metric("mae_test", mae_test)
                mlflow.log_metric("mape_test", mape_test)
                mlflow.log_metric("mse_test", mse_test)
                mlflow.log_metric("rmse_test", rmse_test)

                logging.info(f"Test metrics: MAE: {mae_test}, MAPE: {mape_test}, MSE: {mse_test}, RMSE: {rmse_test}")

                train_metrics = json.dumps(
                    {
                        "mae": mae_train,
                        "mape": mape_train,
                        "mse": mse_train,
                        "rmse": rmse_train
                    },
                    indent=4
                )

                test_metrics = json.dumps(
                    {
                        "mae": mae_test,
                        "mape": mape_test,
                        "mse": mse_test,
                        "rmse": rmse_test
                    },
                    indent=4
                )

                train_metrics_path = self.config.train_metrics
                test_metrics_path = self.config.test_metrics

                with open(train_metrics_path, 'w') as file:
                    file.write(train_metrics)

                with open(test_metrics_path, 'w') as file:
                    file.write(test_metrics)

                logging.info(f"Model metrics are ready. Saved at: {train_metrics_path}, {train_metrics_path}!")

        except Exception as e:
            logging.error(f"Error occurred while getting model metrics!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    model_evaluation_config = config_manager.get_model_evaluation_config()
    model_evaluator = ModelEvaluator(config=model_evaluation_config)
    model_evaluator.get_model_metrics()
