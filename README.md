# Credit Scoring Model

## Problem Statement
The goal of this project is to build a **Credit Scoring Model** that predicts the creditworthiness of individuals based on their historical financial data. This model will help financial institutions assess whether a person is likely to default on a loan, helping them make informed decisions about lending. The model uses various features such as age, gender, income, education level, and more to classify the individual into different creditworthiness categories (Low, Average, High).

## About the Dataset
The dataset used for this project is a **Credit Score Classification Dataset**, which contains various attributes that influence a person's credit score. The data provides insight into how various demographic and financial characteristics correlate with an individual's creditworthiness. The dataset is publicly available on Kaggle and has been used widely for similar credit scoring tasks.

### Dataset Information:
- **Source**: Kaggle (Credit Score Classification Dataset)
- **Rows**: 164
- **Columns**: 8

### Features:
The dataset consists of the following columns:

1. **Age**: The age of the individual (numeric)
2. **Gender**: The gender of the individual (categorical: 'Male' or 'Female')
3. **Income**: The annual income of the individual (numeric)
4. **Education**: The highest education level achieved by the individual (categorical: 'High School Diploma', 'Associate\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'Doctorate')
5. **Marital Status**: The marital status of the individual (categorical: 'Single' or 'Married')
6. **Number of Children**: The number of children the individual has (numeric)
7. **Home Ownership**: The type of housing the individual owns (categorical: 'Rented' or 'Owned')
8. **Credit Score**: The target variable representing the creditworthiness of the individual (categorical: 'Low', 'Average', 'High')

## Why This Dataset?
This dataset is suitable for building a credit scoring model because it includes relevant features like income, education, and home ownership, which are commonly used by financial institutions to evaluate creditworthiness. Additionally, the target variable (Credit Score) has a clear classification that makes it ideal for classification models.

### Benefits of this dataset:
- It provides a broad range of demographic and financial features, which makes it realistic for credit scoring tasks.
- It is relatively small, making it easy to handle and quick to experiment with.
- The dataset has been well-used in various machine learning tutorials, making it a great starting point for classification problems.

## Model
The model used for this project is a **Logistic Regression** classifier, which is a common algorithm for binary and multi-class classification tasks. It works by estimating probabilities of different classes based on the input features. In this case, we classify individuals into three categories of creditworthiness: Low, Average, and High.

### Steps Taken:
1. **Data Preprocessing**:
   - Handled missing values and encoded categorical features (e.g., gender, education level).
   - Scaled numerical features (such as income) to ensure all features are on a similar scale for the model.

2. **Model Training**:
   - Trained the Logistic Regression model on the preprocessed dataset using cross-validation.

3. **Model Evaluation**:
   - Evaluated the model's performance using accuracy, precision, recall, and F1-score metrics.
   - Fine-tuned the model by experimenting with different hyperparameters.

4. **Model Saving**:
   - The trained model was saved as a pickle file (`model.pkl`) for future use, such as making predictions on new data.
