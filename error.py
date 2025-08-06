import pandas as pd
import joblib

# Load dataset
file_path = "/Users/adhithyana/Downloads/vs code/miniproject/training_data.csv"
df = pd.read_csv(file_path)

# Extract feature column names (excluding target variable 'prognosis')
X_columns = df.drop(columns=['prognosis']).columns.tolist()

# Save the feature column names
X_columns_path = "/Users/adhithyana/Downloads/vs code/miniproject/X_columns.pkl"
joblib.dump(X_columns, X_columns_path)

print(f"✅ `X_columns.pkl` has been successfully created and saved at: {X_columns_path}")
