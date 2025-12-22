"""
XAI-CROP: Validation Module (Paper-Aligned)

Implements Algorithm 5 from the research paper.
Evaluates the trained XAI-CROP model on a validation dataset
using RMSE, MAE, and R² metrics.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    df = pd.read_csv("../data/cropds_cleaned.csv")

    district_encoder = joblib.load("../models/district_encoder.pkl")
    season_encoder = joblib.load("../models/season_encoder.pkl")
    crop_encoder = joblib.load("../models/crop_encoder.pkl")

    df["District_Name"] = district_encoder.transform(df["District_Name"])
    df["Season"] = season_encoder.transform(df["Season"])
    df["Crop"] = crop_encoder.transform(df["Crop"])

    # Separate features and target
    validation_features = df.drop("Crop", axis=1)
    validation_target = df["Crop"]

    #validation split
    _, X_val, _, y_val = train_test_split(
        validation_features,
        validation_target,
        test_size=0.2,
        random_state=42,
        stratify=validation_target
    )

    model = joblib.load("../models/xai_crop_decision_tree.pkl")

    predicted_crop_yield = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, predicted_crop_yield))
    mae = mean_absolute_error(y_val, predicted_crop_yield)
    r2 = r2_score(y_val, predicted_crop_yield)

    print("XAI-CROP VALIDATION RESULTS")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"R² Score:                     {r2:.4f}")

    return rmse, mae, r2


if __name__ == "__main__":
    main()
