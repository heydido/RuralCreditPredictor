import sys
import pandas as pd
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.config.configuration import ConfigurationManager
from src.RuralCreditPredictor.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.data = pd.read_csv(self.config.raw_file)

    def get_columns_list(self, log=True) -> tuple:
        try:
            if log:
                logging.info("Getting num/cat columns list:")

            cat_features = list(self.config.cat_features.keys())
            discrete_num_features = list(self.config.discrete_num_features.keys())
            continuous_num_features = list(self.config.continuous_num_features.keys())
            num_features = discrete_num_features + continuous_num_features

            selected_features = list(self.config.selected_features.keys())
            target_variable = list(self.config.target_variable.keys())

            if log:
                logging.info(
                    f"Columns list ready! \n"
                    f" - All Cat features: {cat_features}, \n"
                    f" - All Num features: {num_features}, \n"
                    f" - Selected features: {selected_features}, \n"
                    f" - Target variable: {target_variable}"
                )

            return cat_features, num_features, selected_features, target_variable

        except Exception as e:
            if log:
                logging.error(f"Error occurred while getting columns list!")
            raise CustomException(e, sys)

    def impute_missing_values(self) -> None:
        try:
            logging.info("Imputing missing values:")

            cat_features, num_features, _, _ = self.get_columns_list()

            df = self.data
            all_features = list(df.columns)

            for feature in all_features:
                if df[feature].isnull().sum() > 0:
                    if feature in cat_features:
                        df[feature] = df[feature].fillna(df[feature].mode()[0])  # Mode imputation for cat_features
                        logging.info("Mode imputation applied for: {}".format(feature))

                    elif feature in num_features:
                        df[feature] = df[feature].fillna(df[feature].mean())  # Mean imputation for num_features
                        logging.info("Mean imputation applied for: {}".format(feature))

            logging.info("Imputed missing values successfully!")

            self.data = df

        except Exception as e:
            logging.error(f"Error occurred while imputing missing values!")
            raise CustomException(e, sys)

    def drop_under_and_above_age(self) -> None:
        try:
            logging.info("Dropping data with age <18 and > 65:")

            df = self.data

            under_age, over_age = len(df[df['age'] < 18]), len(df[df['age'] > 65])
            df = df.drop(df[(df['age'] < 18) | (df['age'] > 65)].index)

            logging.info("Dropped {} rows with age <18 and {} rows with age >65".format(under_age, over_age))

            self.data = df

        except Exception as e:
            logging.error(f"Error occurred while dropping data with age <18 and >65!")
            raise CustomException(e, sys)

    def drop_outliers(self) -> None:
        try:
            logging.info("Dropping outliers in continuous numerical features:")

            df = self.data
            continuous_num_features = list(self.config.continuous_num_features.keys())

            for feature in continuous_num_features:
                q1 = df[feature].quantile(0.25)
                q3 = df[feature].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                df_filtered = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
                df = df_filtered

            self.data = df

            logging.info("Dropped outliers in continuous numerical features successfully!")

        except Exception as e:
            logging.error(f"Error occurred while dropping outliers in continuous numerical features!")
            raise CustomException(e, sys)

    def get_preprocessed_data(self) -> None:
        try:
            logging.info("Dropping unused columns and getting preprocessed data:")

            df = self.data

            _, _, selected_features, target_variable = self.get_columns_list(log=False)

            df = df[selected_features + target_variable]

            self.data = df

            logging.info("Dropped unused columns. Preprocessed data ready to export!")

        except Exception as e:
            logging.error(f"Error occurred while getting preprocessed data!")
            raise CustomException(e, sys)

    def save_preprocessed_data(self) -> None:
        try:
            logging.info("Exporting preprocessed data:")

            df = self.data
            df.to_csv(self.config.preprocessed_file, index=False)

            logging.info("Preprocessed data exported successfully!")

        except Exception as e:
            logging.error(f"Error occurred while exporting preprocessed data!")
            raise CustomException(e, sys)


if __name__ == '__main__':
    config_manager = ConfigurationManager()
    data_preprocessing_config = config_manager.get_data_preprocessing_config()
    data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
    data_preprocessing.impute_missing_values()
    data_preprocessing.drop_under_and_above_age()
    data_preprocessing.drop_outliers()
    data_preprocessing.get_preprocessed_data()
    data_preprocessing.save_preprocessed_data()
