import os
import sys
from flask import Flask, render_template, request
from src.RuralCreditPredictor.logger import logging
from src.RuralCreditPredictor.exception import CustomException
from src.RuralCreditPredictor.config.configuration import ConfigurationManager
from src.RuralCreditPredictor.components.prediction import CustomData, Predictor


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# TODO: Fix the training route
@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            logging.info("> Predicting the loan amount:")

            age = int(request.form['age'])
            sex = request.form['sex']
            annual_income = float(request.form['annual_income'])
            monthly_expenses = float(request.form['monthly_expenses'])
            old_dependents = int(request.form['old_dependents'])
            young_dependents = int(request.form['young_dependents'])
            home_ownership = float(request.form['home_ownership'])
            type_of_house = request.form['type_of_house']
            occupants_count = int(request.form['occupants_count'])
            house_area = float(request.form['house_area'])
            loan_tenure = int(request.form['loan_tenure'])
            loan_installments = int(request.form['loan_installments'])

            custom_data = CustomData(
                age=age,
                sex=sex,
                annual_income=annual_income,
                monthly_expenses=monthly_expenses,
                old_dependents=old_dependents,
                young_dependents=young_dependents,
                home_ownership=home_ownership,
                type_of_house=type_of_house,
                occupants_count=occupants_count,
                house_area=house_area,
                loan_tenure=loan_tenure,
                loan_installments=loan_installments
            )

            input_data = custom_data.get_data_as_df()

            # Get Prediction
            config_manager = ConfigurationManager()
            prediction_config = config_manager.get_prediction_config()
            predictor = Predictor(config=prediction_config)
            loan_amount = int(predictor.predict(input_data))

            logging.info(f"Predicted loan amount: {loan_amount}")

            return render_template('results.html', prediction=str(loan_amount))

        except Exception as e:
            logging.error(f"Error in predicting loan amount!")
            raise CustomException(e, sys)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
