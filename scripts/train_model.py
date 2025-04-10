import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
from scripts.data_preprocessing import preprocess_data

# Load data
df = pd.read_csv('data/credit_data.csv')

# Preprocess data
X, y, preprocessor = preprocess_data(df)

# Train the Logistic Regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)

# Save the trained model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved as models/model.pkl")
