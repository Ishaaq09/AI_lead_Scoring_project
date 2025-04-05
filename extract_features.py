import joblib
import json
import numpy as np

# Load model, feature names, and scaler
model, feature_names, scaler = joblib.load('model.pkl')

# Create a sample dictionary with dummy values (e.g., all 0s or means)
sample_input = dict(zip(feature_names, [0] * len(feature_names)))

# Save to JSON
with open('features.json', 'w') as f:
    json.dump(sample_input, f, indent=4)

print("features.json created successfully!")
