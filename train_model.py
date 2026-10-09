import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_dataset(number_of_customers=600, random_state=42):
    """Create a reproducible synthetic telecom customer churn dataset."""
    rng = np.random.default_rng(random_state)

    data = pd.DataFrame({
        "tenure": rng.integers(1, 73, number_of_customers),
        "monthly_charges": rng.uniform(25, 120, number_of_customers).round(2),
        "contract": rng.choice(
            ["Month-to-month", "One year", "Two year"],
            number_of_customers,
            p=[0.60, 0.25, 0.15]
        ),
        "tech_support": rng.choice(
            ["Yes", "No"], number_of_customers, p=[0.35, 0.65]
        ),
        "internet_service": rng.choice(
            ["DSL", "Fiber optic", "No"],
            number_of_customers,
            p=[0.35, 0.50, 0.15]
        ),
        "payment_method": rng.choice(
            ["Electronic check", "Credit card", "Bank transfer"],
            number_of_customers
        )
    })

    # Synthetic target: 1 = CHURN, 0 = NO CHURN.
    churn_score = (
        1.5 * (data["contract"] == "Month-to-month").astype(int)
        + 0.9 * (data["tenure"] < 18).astype(int)
        + 0.7 * (data["monthly_charges"] > 85).astype(int)
        + 0.8 * (data["tech_support"] == "No").astype(int)
        + 0.4 * (data["internet_service"] == "Fiber optic").astype(int)
    )

    churn = (churn_score >= 2.4).astype(int)

    # Add a small amount of label noise for a more realistic demonstration.
    flip = rng.random(number_of_customers) < 0.04
    data["churn"] = np.where(flip, 1 - churn, churn)

    return data


def train_model():
    print("Creating telecom customer churn dataset...")
    data = create_dataset()
    data.to_csv("telecom_churn.csv", index=False)

    print("Dataset created successfully.")
    print("Number of records:", len(data))

    features = [
        "tenure",
        "monthly_charges",
        "contract",
        "tech_support",
        "internet_service",
        "payment_method"
    ]

    X = data[features]
    y = data["churn"]

    numeric_features = ["tenure", "monthly_charges"]
    categorical_features = [
        "contract",
        "tech_support",
        "internet_service",
        "payment_method"
    ]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "telecom_churn_model.pkl")
    print("\nModel saved as telecom_churn_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "model_name": "Logistic Regression",
        "dataset": "Synthetic Telecom Customer Churn"
    }

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    return accuracy


if __name__ == "__main__":
    train_model()
