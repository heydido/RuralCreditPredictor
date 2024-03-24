import sys
import pickle
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

            model_params = self.config.model_params
            model = RandomForestRegressor(**model_params)
            model.fit(x_train, y_train)

            logging.info("Model trained successfully!")

            logging.info("> Saving model:")

            model_path = self.config.model
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)

            logging.info(f"Model saved successfully at: {model_path}!")

        except Exception as e:
            logging.error(f"Could not save model, error occurred while training model!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    model_training_config = config_manager.get_model_training_config()
    model_trainer = ModelTrainer(config=model_training_config)
    model_trainer.train_model()
