import os
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_DIR = "data/raw"
CSV_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def download_dataset():
    """
    Downloads the IBM Telco Customer Churn dataset from a publicly available mirror
    if it is not already present locally.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        
    if not os.path.exists(CSV_PATH):
        print(f"Dataset not found at {CSV_PATH}. Downloading from mirror...")
        try:
            response = requests.get(DATA_URL, timeout=15)
            response.raise_for_status()
            with open(CSV_PATH, 'wb') as f:
                f.write(response.content)
            print("Download completed successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download dataset. Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' "
                f"manually inside 'data/raw/'. Error: {e}"
            )
    else:
        print(f"Dataset already exists at {CSV_PATH}.")

def load_data():
    """
    Loads the raw telecom churn CSV file.
    """
    download_dataset()
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing dataset at {CSV_PATH}.")
    return pd.read_csv(CSV_PATH)

def clean_data(df):
    """
    Cleans the raw dataframe:
    - Converts TotalCharges to numeric, handles blank spaces ' ' by coercing to NaN
      and filling with 0 (since these correspond to new customers with tenure = 0).
    - Drops customerID.
    - Encodes the target 'Churn' column (Yes -> 1, No -> 0).
    """
    df_clean = df.copy()
    
    # 1. Clean TotalCharges
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'])
    # Impute missing values for TotalCharges with 0 (new customers with tenure == 0)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
    
    # 2. Drop customerID
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    # 3. Map target column Churn to binary 0/1
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
    
    return df_clean

def get_feature_lists():
    """
    Returns lists of numerical and categorical feature names.
    """
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    return numerical_features, categorical_features

def get_preprocessor():
    """
    Returns the ColumnTransformer for scaling numerical and encoding categorical features.
    No fitting is done here to prevent data leakage.
    """
    num_cols, cat_cols = get_feature_lists()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    return preprocessor

def split_data(df):
    """
    Splits the cleaned dataset into train and test sets using stratified sampling.
    """
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in DataFrame.")
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Stratified split to maintain churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test
