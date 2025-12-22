"""
XAI-CROP: Final Model Training & Validation Module

Implements a Decision Tree–based crop recommendation model
with:
1) Hyperparameter tuning using GridSearchCV
2) Iteration-wise performance analysis (paper-aligned)
3) Comprehensive evaluation metrics
4) Publication-style plots

This module produces the final model used by the XAI (LIME) pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def main():
    df = pd.read_csv("../data/cropds_cleaned.csv")

    district_encoder = joblib.load("../models/district_encoder.pkl")
    season_encoder = joblib.load("../models/season_encoder.pkl")
    crop_encoder = joblib.load("../models/crop_encoder.pkl")

    df["District_Name"] = district_encoder.transform(df["District_Name"])
    df["Season"] = season_encoder.transform(df["Season"])
    df["Crop"] = crop_encoder.transform(df["Crop"])

    X = df.drop("Crop", axis=1)
    y = df["Crop"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # hyperparameter tuning using GridSearchCV
    param_grid = {
        "max_depth": [None, 10, 15, 20, 25, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_features": [None, "sqrt", "log2"]
    }

    base_dt = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=base_dt,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_dt = grid_search.best_estimator_

    print("\nBest Decision Tree Parameters:")
    print(grid_search.best_params_)

    y_pred = best_dt.predict(X_test)

    best_acc = accuracy_score(y_test, y_pred)
    best_mse = mean_squared_error(y_test, y_pred)
    best_mae = mean_absolute_error(y_test, y_pred)
    best_r2 = r2_score(y_test, y_pred)

    print("\nFinal Model Performance (Tuned DT)")
    print("----------------------------------")
    print(f"Accuracy : {best_acc:.4f}")
    print(f"MSE      : {best_mse:.4f}")
    print(f"MAE      : {best_mae:.4f}")
    print(f"R²       : {best_r2:.4f}")

    # Save final model
    joblib.dump(best_dt, "../models/xai_crop_decision_tree.pkl")

    depths = [3, 5, 7, 10, 12, 15, 20, 25, 30]

    acc_list, mse_list, mae_list, r2_list = [], [], [], []

    print("\nIteration-wise Performance Analysis")
    print("----------------------------------")

    for depth in depths:
        dt_iter = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=best_dt.min_samples_split,
            min_samples_leaf=best_dt.min_samples_leaf,
            criterion=best_dt.criterion,
            max_features=best_dt.max_features,
            random_state=42
        )

        dt_iter.fit(X_train, y_train)
        y_iter_pred = dt_iter.predict(X_test)

        acc = accuracy_score(y_test, y_iter_pred)
        mse = mean_squared_error(y_test, y_iter_pred)
        mae = mean_absolute_error(y_test, y_iter_pred)
        r2 = r2_score(y_test, y_iter_pred)

        acc_list.append(acc)
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

        print(
            f"Depth={depth:>2} | "
            f"Acc={acc:.4f} | "
            f"MSE={mse:.4f} | "
            f"MAE={mae:.4f} | "
            f"R²={r2:.4f}"
        )

    plt.figure(figsize=(10, 6))

    plt.plot(depths, acc_list, marker="o", label="Accuracy")
    plt.plot(depths, mse_list, marker="s", label="MSE")
    plt.plot(depths, mae_list, marker="^", label="MAE")
    plt.plot(depths, r2_list, marker="d", label="R²")

    plt.xlabel("Iteration (Decision Tree Depth)")
    plt.ylabel("Metric Value")
    plt.title("Iteration-wise Performance of XAI-CROP (Decision Tree)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("../results/iteration_vs_performance.png")
    plt.show()

    print("\nTraining, validation, and analysis completed.")
    print("Plot saved as: results/iteration_vs_performance.png")


if __name__ == "__main__":
    main()
