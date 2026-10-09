import json
import os
import unittest

import joblib
import pandas as pd

from train_model import create_dataset


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate the pipeline outputs once before testing.
        from train_model import train_model
        train_model()

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("telecom_churn.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("telecom_churn_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_dataset_has_telecom_columns(self):
        data = pd.read_csv("telecom_churn.csv")
        expected_columns = {
            "tenure",
            "monthly_charges",
            "contract",
            "tech_support",
            "internet_service",
            "payment_method",
            "churn"
        }
        self.assertTrue(expected_columns.issubset(set(data.columns)))
        self.assertEqual(len(data), 600)

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r", encoding="utf-8") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("telecom_churn_model.pkl")
        sample = pd.DataFrame([{
            "tenure": 6,
            "monthly_charges": 95.0,
            "contract": "Month-to-month",
            "tech_support": "No",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check"
        }])

        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])


if __name__ == "__main__":
    unittest.main()
