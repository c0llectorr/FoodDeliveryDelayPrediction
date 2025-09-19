# app/app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Assume folder structure:
# food_delivery_app/
#   app/             <-- this script lives here
#   models/          <-- regression_model.joblib and classification_model.joblib live here
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))

REG_MODEL_PATH = os.path.join(MODELS_DIR, r"time_prediction_model/xgb_regression_model.joblib")
CLF_MODEL_PATH = os.path.join(MODELS_DIR, r"late_delivery_model/xgb_classifier_model.joblib")

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Model load error for {path}: {e}")
        return None

reg_model = safe_load(REG_MODEL_PATH)
clf_model = safe_load(CLF_MODEL_PATH)

# ---- Assumed mappings ----
WEATHER_MAP = {
    "Clear": 1,
    "Cloudy": 2,
    "Rain": 3,
    "Storm": 4,
    "Snow": 5
}

TRAFFIC_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

VEHICLE_MAP = {
    "Bike": 1,
    "Scooter": 2,
    "Motorbike": 3,
    "Car": 4
}

TIME_OF_DAY_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]

SPEED_KMPH = {
    "Bike": 15,
    "Scooter": 20,
    "Motorbike": 25,
    "Car": 30
}

def regressionInput(form):
    distance = float(form.get("Distance_km", 0))
    weather = form.get("Weather", "Clear")
    traffic = form.get("Traffic_Level", "Low")
    time_of_day = form.get("Time_of_Day", "Morning")
    vehicle = form.get("Vehicle_Type", "Bike")
    prep = float(form.get("Preparation_Time_min", 0))
    exp = float(form.get("Courier_Experience_yrs", 0))

    weather_code = WEATHER_MAP.get(weather, 0)
    traffic_code = TRAFFIC_MAP.get(traffic, 0)
    vehicle_code = VEHICLE_MAP.get(vehicle, 0)

    tod_a = 1 if time_of_day == "Afternoon" else 0
    tod_e = 1 if time_of_day == "Evening" else 0
    tod_m = 1 if time_of_day == "Morning" else 0
    tod_n = 1 if time_of_day == "Night" else 0

    features = [distance, weather_code, traffic_code, vehicle_code,
                prep, exp, tod_a, tod_e, tod_m, tod_n]

    return np.array([features], dtype=float)

def classificationInput(form, predicted_delivery_time):
    distance = float(form.get("Distance_km", 0))
    weather = form.get("Weather", "Clear")
    traffic = form.get("Traffic_Level", "Low")
    time_of_day = form.get("Time_of_Day", "Morning")
    vehicle = form.get("Vehicle_Type", "Bike")
    prep = float(form.get("Preparation_Time_min", 0))
    exp = float(form.get("Courier_Experience_yrs", 0))

    weather_code = WEATHER_MAP.get(weather, 0)
    traffic_code = TRAFFIC_MAP.get(traffic, 0)
    vehicle_code = VEHICLE_MAP.get(vehicle, 0)

    tod_a = 1 if time_of_day == "Afternoon" else 0
    tod_e = 1 if time_of_day == "Evening" else 0
    tod_m = 1 if time_of_day == "Morning" else 0
    tod_n = 1 if time_of_day == "Night" else 0

    delivery_time = float(predicted_delivery_time)

    avg_speed = SPEED_KMPH.get(vehicle, 20)
    travel_time_min = (distance / avg_speed) * 60.0 if avg_speed > 0 else 0.0

    base_time = prep + travel_time_min

    buffer = 5.0
    if traffic == "High":
        buffer += 7.0
    elif traffic == "Medium":
        buffer += 3.0
    if weather in ("Rain", "Storm", "Snow"):
        buffer += 5.0

    expected_time = base_time + buffer

    features = [distance, weather_code, traffic_code, vehicle_code,
                prep, exp, delivery_time,
                tod_a, tod_e, tod_m, tod_n,
                base_time, expected_time]

    return np.array([features], dtype=float)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    if reg_model is None:
        return "Regression model not found in models/ folder. Place regression_model.joblib there.", 500

    X_reg = regressionInput(form)
    try:
        reg_pred = reg_model.predict(X_reg)
        reg_pred_val = float(np.array(reg_pred).ravel()[0])
    except Exception as e:
        return f"Regression model prediction error: {e}", 500

    if clf_model is None:
        return render_template("index.html",
                               error_msg="Classification model not found in models/ folder; regression result shown.",
                               regression_result=round(reg_pred_val, 2))

    X_clf = classificationInput(form, reg_pred_val)
    try:
        clf_pred = clf_model.predict(X_clf)
        clf_proba = None
        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba(X_clf)
            clf_proba = float(proba[0][1])
        clf_label = int(np.array(clf_pred).ravel()[0])
    except Exception as e:
        return f"Classification model prediction error: {e}", 500

    result = {
        "predicted_delivery_time_min": round(reg_pred_val, 2),
        "is_late": bool(clf_label),
        "late_probability": round(clf_proba, 3) if clf_proba is not None else None
    }

    return render_template("index.html",
                           regression_result=result["predicted_delivery_time_min"],
                           classification_label=result["is_late"],
                           classification_prob=result["late_probability"])

if __name__ == "__main__":
    app.run(debug=True, port=5050)
