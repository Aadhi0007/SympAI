import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from textblob import TextBlob
import tkinter as tk
from tkinter import messagebox
import logging

# Configure logging to a file instead of the terminal
logging.basicConfig(filename='diagnosis_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error("Dataset file not found.")
        messagebox.showerror("Error", "Dataset file not found! Ensure the CSV is in the correct directory.")
        exit()
    except pd.errors.EmptyDataError:
        logging.error("Dataset file is empty.")
        messagebox.showerror("Error", "Dataset file is empty!")
        exit()
    except pd.errors.ParserError:
        logging.error("Error parsing the dataset.")
        messagebox.showerror("Error", "Error parsing the dataset. Please check the file format.")
        exit()

# Preprocess dataset
def preprocess_data(df):
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Encode categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Encode target labels
    y_encoded = LabelEncoder().fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, scaler, X.columns, LabelEncoder().fit(y).classes_

# Build the model
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Suggest tests based on the predicted disease
def suggest_tests(disease):
    disease_to_tests = {
        'Disease1': ['Blood test', 'X-ray'],
        'Disease2': ['MRI scan', 'ECG'],
        'Disease3': ['Ultrasound', 'CT scan'],
        'Disease4': ['Blood culture', 'Urine test'],
    }
    return disease_to_tests.get(disease, ['No test suggestions available. Please consult a medical professional.'])

# Diagnose function with spelling correction and symptom selection
def diagnose_dl(input_symptoms, model, X_columns, scaler, all_possible_diseases):
    corrected_symptoms = []
    corrections_made = []
    for symptom in input_symptoms:
        corrected_symptom = str(TextBlob(symptom).correct())
        if corrected_symptom != symptom:
            corrections_made.append((symptom, corrected_symptom))
        corrected_symptoms.append(corrected_symptom)

    if corrections_made:
        for original, corrected in corrections_made:
            logging.info(f"Symptom correction: {original} -> {corrected}")

    valid_symptoms = []
    for symptom in corrected_symptoms:
        matches = [col for col in X_columns if symptom.lower() in col.lower()]
        if matches:
            for idx, match in enumerate(matches):
                logging.info(f"Symptom match: {match}")
            valid_symptoms.extend(matches)

    input_data = {symptom: 1 if symptom in valid_symptoms else 0 for symptom in X_columns}
    input_df = pd.DataFrame([input_data])
    input_array = scaler.transform(input_df)

    prediction = model.predict(input_array)
    predicted_probabilities = prediction[0]

    sorted_diseases_with_probabilities = sorted(
        [(disease, prob) for disease, prob in zip(all_possible_diseases, predicted_probabilities) if prob > 0.01],
        key=lambda x: x[1],
        reverse=True
    )

    if sorted_diseases_with_probabilities:
        top_disease, top_probability = sorted_diseases_with_probabilities[0]
    else:
        top_disease, top_probability = "No disease predicted", 0

    test_suggestions = suggest_tests(top_disease)

    return sorted_diseases_with_probabilities, top_disease, top_probability, test_suggestions

# GUI for user interaction
class SymptomDiagnosisApp:
    def __init__(self, root, model, X_columns, scaler, all_possible_diseases):
        self.root = root
        self.model = model
        self.X_columns = X_columns
        self.scaler = scaler
        self.all_possible_diseases = all_possible_diseases

        self.root.title("Disease Diagnosis System")
        self.label = tk.Label(root, text="Enter the number of symptoms:", font=("Helvetica", 14))
        self.label.pack(pady=10)
        self.num_symptoms_entry = tk.Entry(root, width=10)
        self.num_symptoms_entry.pack(pady=5)
        self.submit_num_button = tk.Button(root, text="Submit Number of Symptoms", command=self.submit_num_symptoms)
        self.submit_num_button.pack(pady=10)
        self.symptom_entries = []

    def submit_num_symptoms(self):
        try:
            num_symptoms = int(self.num_symptoms_entry.get().strip())
            if num_symptoms <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Input Error", "Please enter a valid number of symptoms.")
            return

        self.num_symptoms_entry.destroy()
        self.submit_num_button.destroy()

        for i in range(num_symptoms):
            label = tk.Label(self.root, text=f"Enter symptom {i+1}:", font=("Helvetica", 12))
            label.pack(pady=5)
            entry = tk.Entry(self.root, width=50)
            entry.pack(pady=5)
            self.symptom_entries.append(entry)

        self.submit_symptoms_button = tk.Button(self.root, text="Submit Symptoms", command=self.submit_symptoms)
        self.submit_symptoms_button.pack(pady=10)

    def submit_symptoms(self):
        input_symptoms = [entry.get().strip() for entry in self.symptom_entries if entry.get().strip()]
        if not input_symptoms:
            messagebox.showwarning("Input Error", "Please enter the symptoms.")
            return

        results, top_disease, top_probability, test_suggestions = diagnose_dl(
            input_symptoms, self.model, self.X_columns, self.scaler, self.all_possible_diseases
        )

        result_message = "Predicted Diseases and Probabilities:\n"
        for disease, probability in results:
            result_message += f"{disease}: {probability * 100:.2f}% chance\n"

        result_message += f"\nTop Prediction: {top_disease} ({top_probability * 100:.2f}% confidence)\n"
        result_message += "\nSuggested Tests:\n" + "\n".join(test_suggestions)

        messagebox.showinfo("Diagnosis Results", result_message)

# Main function to run the application without terminal
if __name__ == "__main__":
    dataset_path = 'data1.csv'
    df = load_dataset(dataset_path)

    X_scaled, y_encoded, scaler, X_columns, all_possible_diseases = preprocess_data(df)

    model = build_model(X_scaled.shape[1], len(all_possible_diseases))

    # Fit the modelclear
    model.fit(X_scaled, y_encoded, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

    # Start the Tkinter GUI
    root = tk.Tk()
    app = SymptomDiagnosisApp(root, model, X_columns, scaler, all_possible_diseases)
    root.mainloop()
