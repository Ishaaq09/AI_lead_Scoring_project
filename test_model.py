import pandas as pd
import joblib
import numpy as np

def test_model_loading():
    try:
        model = joblib.load("logistic_model.joblib")
        print("Model loaded successfully.")
    except Exception as e:
        raise AssertionError(f"Failed to load model: {e}")

def test_model_accuracy():
    model = joblib.load("logistic_model.joblib")
    X_test = joblib.load("X_test.joblib")
    y_test = joblib.load("y_test.joblib")
    
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.2f}")
    assert acc > 0.7, "Accuracy too low."

if __name__ == "__main__":
    test_model_loading()
    test_model_accuracy()
