import os
import joblib
import pandas
from dotenv import load_dotenv
from flask import Flask, render_template, request

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

MODEL_DIR = os.path.dirname(MODEL_PATH)

knn_model = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
dt_model = joblib.load(os.path.join(MODEL_DIR, "dt.pkl"))
svc_model = joblib.load(os.path.join(MODEL_DIR, "svc.pkl"))
mlp_model = joblib.load(os.path.join(MODEL_DIR, "mlp.pkl"))

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
    inputData = {}
    for col in columns:
        val = request.form.get(col)
        try:
            if "." in val:
                inputData[col] = float(val)
            else:
                inputData[col] = int(val)
        except (ValueError, TypeError):
            inputData[col] = val
    input_df = pandas.DataFrame([inputData])
    pred_knn = int(knn_model.predict(input_df)[0])
    pred_dt = int(dt_model.predict(input_df)[0])
    pred_svc = int(svc_model.predict(input_df)[0])
    pred_mlp = int(mlp_model.predict(input_df)[0])
    votes = pred_knn + pred_dt + pred_svc + pred_mlp
    final_prediction = 1 if votes >= 2 else 0
    model_preds = {
        "KNN": pred_knn,
        "Decision Tree": pred_dt,
        "SVC": pred_svc,
        "MLP": pred_mlp
    }
    return render_template("index.html", data=inputData, prediction=final_prediction, model_preds=model_preds)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
