# app.py
from flask import Flask, request, jsonify
import pandas as pd
from model import model  # Make sure model.py exports the trained model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        return jsonify({
            'prediction': int(prediction),
            'conversion_probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
