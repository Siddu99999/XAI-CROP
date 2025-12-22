"""
XAI-CROP: Standalone LIME Explanation Script

Generates a LIME explanation for a dataset instance.
Useful for validation and report screenshots.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer


def main():
    df = pd.read_csv("../data/cropds_cleaned.csv")

    district_encoder = joblib.load("../models/district_encoder.pkl")
    season_encoder = joblib.load("../models/season_encoder.pkl")
    crop_encoder = joblib.load("../models/crop_encoder.pkl")
    model = joblib.load("../models/xai_crop_decision_tree.pkl")

    df["District_Name"] = district_encoder.transform(df["District_Name"])
    df["Season"] = season_encoder.transform(df["Season"])
    df["Crop"] = crop_encoder.transform(df["Crop"])

    X = df.drop("Crop", axis=1)

    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=crop_encoder.classes_.tolist(),
        mode="classification",
        discretize_continuous=True
    )

    instance = X.iloc[10].values
    pred_class = int(model.predict([instance])[0])
    pred_crop = crop_encoder.inverse_transform([pred_class])[0]

    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        labels=[pred_class],
        num_features=4
    )

    explanation.save_to_file("lime_explanation.html")
    print(f"LIME explanation generated for crop: {pred_crop}")


if __name__ == "__main__":
    main()
