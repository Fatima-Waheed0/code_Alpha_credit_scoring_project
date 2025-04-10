import pickle
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from scripts.data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv('data/credit_data.csv')
X, y, _ = preprocess_data(df)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
