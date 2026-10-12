import json
import logging
import math
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Load saved model and preprocessing pipeline (same files created by the ML script)
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# Optional: test-set scores shown on the page. Create metrics.json like:
# {"model": "Random Forest Regressor", "mae": 31000, "rmse": 48000, "r2": 0.82}
try:
    with open("metrics.json") as f:
        METRICS = json.load(f)
except (OSError, ValueError):
    METRICS = None

# Approximate rate. Can be changed without editing code: set USD_TO_INR env variable.
USD_TO_INR = float(os.environ.get("USD_TO_INR", 88.0))

OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

# field name -> (label, minimum, maximum, must be > 0 instead of >= 0)
NUMERIC_FIELDS = {
    "longitude": ("Longitude", -125.0, -113.0, False),
    "latitude": ("Latitude", 32.0, 42.5, False),
    "housing_median_age": ("Housing median age", 0.0, 100.0, False),
    "total_rooms": ("Total rooms", 0.0, 100000.0, True),
    "total_bedrooms": ("Total bedrooms", 0.0, 100000.0, False),
    "population": ("Population", 0.0, 100000.0, True),
    "households": ("Households", 0.0, 100000.0, True),
    "median_income": ("Median income", 0.0, 20.0, True),
}

DEFAULTS = {
    "longitude": "-122.0",
    "latitude": "37.0",
    "housing_median_age": "25",
    "total_rooms": "2000",
    "total_bedrooms": "400",
    "population": "1000",
    "households": "350",
    "median_income": "4.0",
    "ocean_proximity": "<1H OCEAN",
}


def parse_and_validate(form_data):
    """Convert form values to numbers and validate them. Raises ValueError with a clear message."""
    data = {}

    for name, (label, low, high, strictly_positive) in NUMERIC_FIELDS.items():
        raw = (form_data.get(name) or "").strip()
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{label} must be a number.")

        # float() accepts "nan" and "inf", so check explicitly
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a valid number.")

        if strictly_positive and value <= 0:
            raise ValueError(f"{label} must be greater than 0.")

        if not low <= value <= high:
            raise ValueError(f"{label} must be between {low:g} and {high:g}.")

        data[name] = value

    if data["total_bedrooms"] > data["total_rooms"]:
        raise ValueError("Total bedrooms cannot be more than total rooms.")

    if data["households"] > data["population"]:
        raise ValueError("Households cannot be more than population.")

    ocean = form_data.get("ocean_proximity", "")
    if ocean not in OCEAN_OPTIONS:
        raise ValueError("Please select a valid location type.")
    data["ocean_proximity"] = ocean

    return data


@app.route("/", methods=["GET", "POST"])
def home():
    dollars = None
    rupees = None
    low = None
    high = None
    error = None
    form_data = dict(DEFAULTS)

    if request.method == "POST":
        # Keep exactly what the user typed
        for key in form_data:
            form_data[key] = request.form.get(key, form_data[key])

        try:
            data = parse_and_validate(form_data)

            # Same column names as the training data
            input_df = pd.DataFrame([data])

            transformed = pipeline.transform(input_df)
            dollars = float(model.predict(transformed)[0])
            rupees = dollars * USD_TO_INR

            # Spread of the individual trees gives a rough range (Random Forest only)
            if hasattr(model, "estimators_"):
                tree_preds = np.array([t.predict(transformed)[0] for t in model.estimators_])
                low, high = (float(v) for v in np.percentile(tree_preds, [10, 90]))

            app.logger.info("Prediction: %.2f", dollars)

        except ValueError as e:
            error = str(e)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Something went wrong while predicting. Please try again."

    return render_template(
        "index.html",
        dollars=dollars,
        rupees=rupees,
        low=low,
        high=high,
        metrics=METRICS,
        usd_to_inr=USD_TO_INR,
        error=error,
        form_data=form_data,
        ocean_options=OCEAN_OPTIONS,
    )


if __name__ == "__main__":
    # Debug only when FLASK_DEBUG=1 (local use). Never enable it on a public server.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")