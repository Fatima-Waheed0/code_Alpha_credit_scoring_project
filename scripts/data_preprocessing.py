import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    # Separate features and target variable
    X = df.drop('Credit Score', axis=1)
    y = df['Credit Score']
    
    # Define categorical and numerical columns
    cat_features = ['Gender', 'Education', 'Marital Status', 'Home Ownership']
    num_features = ['Age', 'Income', 'Number of Children']
    
    # Set up the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), cat_features),
            ('num', StandardScaler(), num_features)
        ])
    
    # Apply preprocessing to data
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor
