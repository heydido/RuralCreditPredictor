import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.processed_data = pd.read_csv(self.config.preprocessed_file)

    def split_data(self, log=True) -> tuple:
        try:
            if log:
                logging.info("> Splitting data into train and test sets:")

            df = self.processed_data

            x, y = df.drop('loan_amount', axis=1), df['loan_amount']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

            if log:
                logging.info("Data splited into train/test successfully!")

            return x_train, x_test, y_train, y_test

        except Exception as e:
            if log:
                logging.error(f"Error occurred while splitting data!")
            raise CustomException(e, sys)

    def get_final_columns_list(self, log=True) -> tuple:
        try:
            if log:
                logging.info("> Getting final columns list:")

            cat_features = list(self.config.cat_features.keys())
            discrete_num_features = list(self.config.discrete_num_features.keys())
            continuous_num_features = list(self.config.continuous_num_features.keys())
            num_features = discrete_num_features + continuous_num_features

            selected_features = list(self.config.selected_features.keys())

            selected_num_features = [feature for feature in selected_features if feature in num_features]
            selected_cat_features = [feature for feature in selected_features if feature in cat_features]

            if log:
                logging.info("Final columns list is ready!")

            return selected_num_features, selected_cat_features

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting final columns list!")
            raise CustomException(e, sys)

    def get_data_transformer(self, log=True) -> ColumnTransformer:
        try:
            if log:
                logging.info("> Getting data transformer:")

            num_transformer = StandardScaler()
            cat_transformer = OneHotEncoder(drop='first')

            selected_num_features, selected_cat_features = self.get_final_columns_list(log=False)

            data_transformer = ColumnTransformer(
                [
                    ("OneHotEncoder", cat_transformer, selected_cat_features),
                    ("StandardScaler", num_transformer, selected_num_features),
                ]
            )

            if log:
                logging.info("Data transformer is ready!")

            return data_transformer

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting data transformer!")
            raise CustomException(e, sys)

    def get_transformed_data(self, log=True) -> tuple:
        try:
            if log:
                logging.info("> Transforming data:")

            _x_train, _x_test, y_train, y_test = self.split_data(log=False)
            data_transformer = self.get_data_transformer(log=False)

            x_train = data_transformer.fit_transform(_x_train)
            x_train = pd.DataFrame(x_train, columns=data_transformer.get_feature_names_out())

            x_test = data_transformer.transform(_x_test)
            x_test = pd.DataFrame(x_test, columns=data_transformer.get_feature_names_out())

            if log:
                logging.info("Data transformed successfully!")

            return data_transformer, x_train, x_test, y_train, y_test

        except Exception as e:
            if log:
                logging.error(f"> Error occurred while transforming data!")
            raise CustomException(e, sys)

    def save_data_transformer(self) -> None:
        try:
            logging.info("> Saving data transformer:")

            data_transformer, _, _, _, _ = self.get_transformed_data(log=False)
            data_transformer_path = self.config.data_transformer

            with open(data_transformer_path, 'wb') as file:
                pickle.dump(data_transformer, file)

            logging.info("Data transformer saved successfully!")

        except Exception as e:
            logging.error(f"Error occurred while saving data transformer!")
            raise CustomException(e, sys)
