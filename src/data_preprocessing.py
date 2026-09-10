"""
Data Preprocessing and Feature Engineering Pipeline
Project: Interpretable Student Academic Risk Prediction
Course: CSE2267 - Machine Learning Techniques
Institution: Presidency University, Bengaluru
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load the raw UCI Student Performance dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path, sep=';')
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct domain-specific features for student academic risk detection.
    - Parent education average
    - Study to free-time ratio
    - High alcohol consumption indicator
    - Absence risk categories
    """
    df = df.copy()

    # 1. Parent education index
    df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2.0

    # 2. Study-to-free-time ratio
    # Add small epsilon to avoid zero division
    df['study_freetime_ratio'] = df['studytime'] / (df['freetime'] + 0.5)

    # 3. Combined weekly alcohol consumption index (Dalc weighted + Walc)
    df['alcohol_index'] = (df['Dalc'] * 5 + df['Walc'] * 2) / 7.0

    # 4. Absence severity indicator
    df['high_absences'] = (df['absences'] > 10).astype(int)

    # 5. Multiple past failures indicator
    df['has_prior_failures'] = (df['failures'] > 0).astype(int)

    return df


def preprocess_and_split(
    raw_data_path: str,
    output_dir: str,
    test_size: float = 0.20,
    random_state: int = 42,
    include_intermediate_grades: bool = False
):
    """
    Full pipeline to clean, encode, transform, and export train/test splits.
    
    Parameters:
    - raw_data_path: path to student-mat.csv
    - output_dir: path to directory where train.csv and test.csv will be saved
    - test_size: test set proportion (default: 0.20)
    - random_state: seed for reproducibility (default: 42)
    - include_intermediate_grades: whether to include G1 & G2 in feature set.
      Default False for true early-warning prediction prior to exams.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_raw_data(raw_data_path)

    # Define binary ground-truth target
    df['At_Risk'] = (df['G3'] < 10).astype(int)

    # Engineer academic and behavioral features
    df = engineer_features(df)

    # Determine features to exclude
    drop_cols = ['G3', 'At_Risk']
    if not include_intermediate_grades:
        drop_cols.extend(['G1', 'G2'])

    X = df.drop(columns=drop_cols)
    y = df['At_Risk']

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Identify numerical and categorical columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    # Fit on train, transform on both train and test to prevent data leakage
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Retrieve feature names after one-hot encoding
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_cat_names)

    # Build clean DataFrames
    train_df = pd.DataFrame(X_train_trans, columns=all_feature_names, index=X_train.index)
    train_df['At_Risk'] = y_train.values

    test_df = pd.DataFrame(X_test_trans, columns=all_feature_names, index=X_test.index)
    test_df['At_Risk'] = y_test.values

    # Export to processed folder
    train_out = os.path.join(output_dir, "train.csv")
    test_out = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Data preprocessing complete!")
    print(f"Training set: {train_df.shape[0]} samples, {len(all_feature_names)} features -> {train_out}")
    print(f"Testing set:  {test_df.shape[0]} samples, {len(all_feature_names)} features -> {test_out}")
    print(f"Class Balance (Train): {train_df['At_Risk'].value_counts().to_dict()}")
    print(f"Class Balance (Test):  {test_df['At_Risk'].value_counts().to_dict()}")

    return train_df, test_df


if __name__ == "__main__":
    base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_project_dir, "data", "student-mat.csv")
    proc_dir = os.path.join(base_project_dir, "data", "processed")

    preprocess_and_split(raw_path, proc_dir)
