from flask import Flask, request, jsonify
import logging
from preprocessing import preprocess_data
from models import (
    load_models,
    predict_accepted_rejected,
    predict_grade,
    predict_subgrade,
    predict_int_rate,
)

app = Flask(__name__)


models = load_models()


@app.route("/")
def hello():
    return "Welcome to the Lending Club API"


@app.route("/predict/loan-approval", methods=["POST"])
def predict_loan_status():
    try:
        data = request.json
        df_processed = preprocess_data(data, models, False)
        is_accepted = predict_accepted_rejected(df_processed, models)

        if is_accepted:
            return jsonify({"Loan Status": "Accepted", "Detail Required": True})
        return jsonify({"Loan Status": "Rejected"})
    except Exception as e:
        logging.error(f"Error during loan approval prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500


@app.route("/predict/grade-and-rate", methods=["POST"])
def predict_detailed():
    try:
        data = request.json

        df_processed = preprocess_data(data, models, True)

        grade = predict_grade(df_processed, models)
        subgrade = predict_subgrade(df_processed, models)
        int_rate = predict_int_rate(df_processed, models)

        return jsonify(
            {
                "Loan Status": "Accepted",
                "Grade": grade,
                "Subgrade": subgrade,
                "Interest Rate": int_rate,
            }
        )
    except Exception as e:
        logging.error(f"Error during grade/interest rate prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500


if __name__ == "__main__":
    app.run(debug=True)
