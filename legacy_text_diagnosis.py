import re
import json
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the file path correctly as a string (use absolute path if necessary)
file_path = "/Users/adhithyana/Downloads/vs code/miniproject/training_data.csv"  # Update with your correct path

# Check if the file exists before proceeding
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Load dataset (CSV file containing symptoms and corresponding diseases)
df = pd.read_csv(file_path)

# Print the column names to check if 'symptoms' exists
print("Columns in the dataset:", df.columns)

# Combine all symptom columns into a list (excluding 'prognosis' column)
symptom_columns = df.columns[:-1]  # Excluding 'prognosis' column
df['symptoms'] = df[symptom_columns].apply(lambda row: ' '.join([col for col, val in row.items() if val == 1]), axis=1)

# Preprocessing function
def preprocess_text(text):
    """Cleans and preprocesses medical report text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Preprocess dataset
df["symptoms"] = df["symptoms"].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
df['prognosis'] = label_encoder.fit_transform(df['prognosis'])

# Convert text data into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["symptoms"]).toarray()
y = df["prognosis"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Deep Learning Model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {test_accuracy * 100:.2f}%")

# Function to predict disease from user input
def analyze_medical_report(report):
    """Predicts possible diseases based on input symptoms."""
    cleaned_text = preprocess_text(report)
    transformed_text = vectorizer.transform([cleaned_text]).toarray()
    predicted_index = model.predict(transformed_text).argmax()
    predicted_disease = label_encoder.inverse_transform([predicted_index])[0]
    return {"Extracted Symptoms": report, "Predicted Disease": predicted_disease}

# Collect symptoms from the user
input_medical_report = []
n = int(input("Enter the number of symptoms: "))
for i in range(n):
    symptom = input(f"Enter symptom {i + 1}: ")
    input_medical_report.append(symptom)

# Combine the symptoms into a single string for prediction
medical_report = " ".join(input_medical_report)

# Get the prediction based on the entered symptoms
result = analyze_medical_report(medical_report)

# Output the result in JSON format
print(json.dumps(result, indent=4))
