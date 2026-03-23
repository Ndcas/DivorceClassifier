import os
import joblib
import pandas
from dotenv import load_dotenv
from flask import Flask, render_template, request

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

model = joblib.load(MODEL_PATH)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    columns = [
        "age_at_marriage",
        "marriage_duration_years",
        "num_children",
        "education_level",
        "employment_status",
        "combined_income",
        "religious_compatibility",
        "cultural_background_match",
        "communication_score",
        "conflict_frequency",
        "conflict_resolution_style",
        "financial_stress_level",
        "mental_health_issues",
        "infidelity_occurred",
        "counseling_attended",
        "social_support",
        "shared_hobbies_count",
        "marriage_type",
        "pre_marital_cohabitation",
        "domestic_violence_history",
        "trust_score"
    ]
    input_data = {}
    for col in columns:
        val = request.form.get(col)
        try:
            if "." in val:
                input_data[col] = float(val)
            else:
                input_data[col] = int(val)
        except (ValueError, TypeError):
            input_data[col] = val
    input_df = pandas.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
