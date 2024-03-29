import os
import sys

import mlflow
import dagshub
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestRegressor
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.entity.config_entity import ModelTrainingConfig
from src.RuralCreditPredictor.config.configuration import ConfigurationManager
from src.RuralCreditPredictor.components.data_transformation import DataTransformation


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    @staticmethod
    def get_data(log=True):
        try:
            if log:
                logging.info("> Getting data for model training:")

            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformer_config(log=False)
            transformer = DataTransformation(config=data_transformation_config)
            _, x_train, x_test, y_train, y_test = transformer.get_transformed_data(log=False)

            if log:
                logging.info("Data is ready for model training!")

            return x_train, x_test, y_train, y_test

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data for model training!")
            raise CustomException(e, sys)

    def train_model(self):
        try:
            logging.info("> Training model:")

            x_train, x_test, y_train, y_test = self.get_data()

            # Initialize DagsHub
            dagshub.init("RuralCreditPredictor", "heydido", mlflow=True)

            # Set the experiment name
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run() as run:
                # All the parameters are logged in mlflow
                model_params = self.config.model_params

                mlflow.log_param("criterion", model_params.criterion)
                mlflow.log_param("max_depth", model_params.max_depth)
                mlflow.log_param("max_features", model_params.max_features)
                mlflow.log_param("min_samples_leaf", model_params.min_samples_leaf)
                mlflow.log_param("min_samples_split", model_params.min_samples_split)
                mlflow.log_param("n_estimators", model_params.n_estimators)
                mlflow.log_param("random_state", model_params.random_state)

                logging.info("Logged model parameters successfully! Training Started....")

                rfr = RandomForestRegressor(**model_params, verbose=2, n_jobs=-1)
                rfr.fit(x_train, y_train)

                logging.info("Model trained successfully!")

                # Comment below two lines to run experiment/save model locally
                remote_server_uri = "https://dagshub.com/heydido/RuralCreditPredictor.mlflow"
                mlflow.set_tracking_uri(remote_server_uri)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    logging.info("> Saving model - Mode: Remote")

                    mlflow.sklearn.log_model(rfr, "model", registered_model_name="RandomForestModel")

                    model_uri = mlflow.get_artifact_uri("model")

                    logging.info(f"Model saved successfully at: {model_uri}")

                else:
                    logging.info("> Saving model - Mode: Local")

                    mlflow.sklearn.log_model(rfr, "model")

                    run_id = run.info.run_id
                    experiment_id = mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id
                    model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"

                    logging.info(f"Model saved successfully at: {model_path}!")

                # Save run id to track evaluation metrics
                logging.info("> Saving run ID:")

                run_id = run.info.run_id

                run_id_path = os.path.join(self.config.root_dir, "latest_run_id.txt")
                with open(run_id_path, "w") as f:
                    f.write(run_id)

                logging.info(f"Run ID saved successfully at: {run_id_path}")

                experiment_id = mlflow.get_experiment_by_name(self.config.experiment_name).experiment_id
                model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"

                logging.info(f"Model saved successfully at: {model_path}!")

        except Exception as e:
            logging.error(f"Could not save model, error occurred while training model")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    model_training_config = config_manager.get_model_training_config()
    model_trainer = ModelTrainer(config=model_training_config)
    model_trainer.train_model()
