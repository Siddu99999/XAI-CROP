"""
XAI-CROP: Web Application Backend

Takes farmer input, predicts the best crop,
and generates a human-readable LIME explanation.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from lime.lime_tabular import LimeTabularExplainer

# --------------------------------------------------
# Flask initialization
# --------------------------------------------------
app = Flask(__name__)

model = joblib.load("../models/xai_crop_decision_tree.pkl")
district_encoder = joblib.load("../models/district_encoder.pkl")
season_encoder = joblib.load("../models/season_encoder.pkl")
crop_encoder = joblib.load("../models/crop_encoder.pkl")

df = pd.read_csv("../data/cropds_cleaned.csv")
df["District_Name"] = district_encoder.transform(df["District_Name"])
df["Season"] = season_encoder.transform(df["Season"])
df["Crop"] = crop_encoder.transform(df["Crop"])

X = df.drop("Crop", axis=1)
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=crop_encoder.classes_.tolist(),
    mode="classification",
    discretize_continuous=True,
    kernel_width=1.5,
    verbose=True
)

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # ---- Read & normalize input ----
        district = request.form["district"].strip().upper()
        season = request.form["season"].strip().lower()
        area = float(request.form["area"])
        production = float(request.form["production"])

        # ---- Encode input ----
        district_enc = district_encoder.transform([district])[0]
        season_enc = season_encoder.transform([season])[0]
        instance = np.array([[district_enc, season_enc, area, production]])

        # ---- Predict crop ----
        pred_class = int(model.predict(instance)[0])
        pred_crop = crop_encoder.inverse_transform([pred_class])[0]

        # ---- Natural language explanation ----
        nl_explanation = (
            f"The system recommends cultivating {pred_crop} because the "
            f"given season ({season}), district ({district}), land area ({area}), "
            f"and expected production ({production}) closely resemble historical "
            f"conditions where this crop performed well."
        )

        # ---- LIME explanation (FOR THE PREDICTED CLASS) ----
        explanation = explainer.explain_instance(
            instance.flatten(),
            model.predict_proba,
            labels=[pred_class],
            num_features=4
        )

        # ---- Build LIME HTML (human-friendly) ----
        lime_html = explanation.as_html()

        final_lime_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAI-CROP – Why {pred_crop}?</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 30px;
                    background-color: #f4f6f5;
                }}
                .header {{
                    background-color: #2e7d32;
                    color: white;
                    padding: 20px;
                    border-radius: 6px;
                }}
                .section {{
                    background-color: white;
                    margin-top: 25px;
                    padding: 20px;
                    border-left: 6px solid #2e7d32;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>

        <div class="header">
            <h1>XAI-CROP: Explanation of Recommendation</h1>
            <p>This page explains why <b>{pred_crop}</b> was recommended.</p>
        </div>

        <div class="section">
            <h2>What does LIME do?</h2>
            <ul>
                <li>Generates small variations of the farmer’s input</li>
                <li>Observes model predictions for these variations</li>
                <li>Fits a local linear surrogate model</li>
                <li>Uses feature weights as explanations</li>
            </ul>
        </div>

        <div class="section">
            <h2>Feature Contribution (Explaining {pred_crop})</h2>
            <p>
                Green bars support the recommendation of {pred_crop}.
                Red bars oppose it. Longer bars mean stronger influence.
            </p>
            {lime_html}
        </div>

        </body>
        </html>
        """

        # ---- Save LIME HTML in static folder ----
        lime_path = os.path.join("static", "lime_explanation.html")
        with open(lime_path, "w", encoding="utf-8") as f:
            f.write(final_lime_html)

        return render_template(
            "result.html",
            crop=pred_crop,
            explanation=nl_explanation
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
