from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from models import train_diabetes, train_standard, load_model_and_scaler

app = Flask(__name__)
DATA_DIR = "data"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train_all")
def train_all():

    results = {}

    try:
        results["diabetes"] = train_diabetes(os.path.join(DATA_DIR, "diabetes.csv"))
    except Exception as e:
        results["diabetes"] = str(e)

    try:
        results["heart"] = train_standard(
            os.path.join(DATA_DIR, "heart_disease.csv"),
            "TenYearCHD",
            "Heart"
        )
    except Exception as e:
        results["heart"] = str(e)

    try:
        cancer_path = os.path.join(DATA_DIR, "breast_cancer.csv")
        df = pd.read_csv(cancer_path)
        tcol = "target" if "target" in df.columns else "diagnosis"
        results["cancer"] = train_standard(cancer_path, tcol, "Cancer", drop_cols=["id"])
    except Exception as e:
        results["cancer"] = str(e)

    return jsonify(results)


@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    import numpy as np
    import pandas as pd

    model, scaler = load_model_and_scaler(disease)

    if model is None:
        return jsonify({"error": "Model not trained"}), 400

    data = request.json

    csv_map = {
        "Diabetes": "diabetes.csv",
        "Heart": "heart_disease.csv",
        "Cancer": "breast_cancer.csv"
    }

    df = pd.read_csv(os.path.join(DATA_DIR, csv_map[disease]))

    feature_names = [
        col for col in df.columns
        if col not in ("Outcome", "TenYearCHD", "target", "diagnosis", "id")
    ]

    # Collect features in correct order
    values = [float(data.get(f, 0)) for f in feature_names]

    # FIX: scaler expects DataFrame with feature names
    X_df = pd.DataFrame([values], columns=feature_names)
    X_scaled = scaler.transform(X_df)

    pred = int(model.predict(X_scaled)[0])

    try:
        proba = model.predict_proba(X_scaled).tolist()[0]
    except:
        proba = None

    return jsonify({"prediction": pred, "proba": proba})


# ROUTES
@app.route("/diabetes")
def diabetes_page():
    return render_template("diabetes.html")


@app.route("/heart")
def heart_page():
    return render_template("heart.html")


@app.route("/cancer")
def cancer_page():
    return render_template("cancer.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/patientdashboard")
def patient_page():
    return render_template("patientdasboard.html")


if __name__ == "__main__":
    app.run(debug=True)
