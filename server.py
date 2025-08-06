import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Load the trained model
model = tf.keras.models.load_model("/Users/adhithyana/Downloads/vs code/miniproject/disease_model.h5")

# ✅ Load the scaler & feature columns
scaler = joblib.load("/Users/adhithyana/Downloads/vs code/miniproject/scaler.pkl")
X_columns = joblib.load("/Users/adhithyana/Downloads/vs code/miniproject/X_columns.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Get user input (symptoms) from JSON request
        data = request.get_json()
        input_symptoms = data.get("symptoms", [])

        # ✅ Create an empty feature dictionary with all symptoms set to 0
        input_data = {col: 0 for col in X_columns}

        # ✅ Activate relevant symptom features (set to 1)
        for symptom in input_symptoms:
            if symptom in input_data:
                input_data[symptom] = 1

        # ✅ Convert to DataFrame (to match StandardScaler expectations)
        input_df = pd.DataFrame([input_data])

        # ✅ Apply StandardScaler transformation (ensuring column names match)
        input_scaled = scaler.transform(input_df)

        # ✅ Check if input feature size matches model expectation
        if input_scaled.shape[1] != model.input_shape[1]:
            return jsonify({"error": f"Feature mismatch: Model expects {model.input_shape[1]} features, but received {input_scaled.shape[1]}"}), 400

        # ✅ Make Prediction
        prediction = model.predict(input_scaled)
        predicted_index = np.argmax(prediction, axis=1)[0]

        # ✅ Load label encoder and decode predicted disease
        label_encoder = joblib.load("/Users/adhithyana/Downloads/vs code/miniproject/label_encoders.pkl")
        predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

        # ✅ Return prediction as JSON
        return jsonify({
            "Predicted Disease": predicted_disease,
            "Confidence Score": f"{max(prediction[0]) * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
