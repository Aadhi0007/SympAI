import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path


# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Project directory
BASE_DIR = Path(__file__).resolve().parent


# Load trained model and preprocessing components
MODEL_PATH = BASE_DIR / "disease_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"
X_COLUMNS_PATH = BASE_DIR / "X_columns.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoders.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_columns = joblib.load(X_COLUMNS_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input (symptoms) from JSON request
        data = request.get_json()
        input_symptoms = data.get("symptoms", [])

        # Create an empty feature dictionary with all symptoms set to 0
        input_data = {col: 0 for col in X_columns}

        # Activate relevant symptom features
        for symptom in input_symptoms:
            if symptom in input_data:
                input_data[symptom] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply StandardScaler transformation
        input_scaled = scaler.transform(input_df)

        # Check feature size
        if input_scaled.shape[1] != model.input_shape[1]:
            return jsonify({
                "error": (
                    f"Feature mismatch: Model expects "
                    f"{model.input_shape[1]} features, "
                    f"but received {input_scaled.shape[1]}"
                )
            }), 400

        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        predicted_index = np.argmax(prediction, axis=1)[0]

        # Decode predicted disease
        predicted_disease = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # Return prediction
        return jsonify({
            "Predicted Disease": predicted_disease,
            "Confidence Score": f"{max(prediction[0]) * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
