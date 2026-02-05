import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any

def preprocess_titanic(
    data_dir: Path, 
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocesses the Titanic dataset from 'train.csv'.
    
    Steps:
    1. Load train.csv (ignoring test.csv as it lacks labels for benchmarking).
    2. Missing Values: Age (median), Cabin (feature engineered), Embarked (mode).
    3. Encoding: Sex (binary), Embarked (One-Hot).
    4. Scaling: StandardScaler on numericals.
    5. Split: Train/Val/Test (default 70/15/15).
    """
    file_path = data_dir / "train.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Titanic train.csv not found at {file_path}")

    df = pd.read_csv(file_path)
    
    # --- 1. Handle Missing Values ---
    # Age: Fill with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: Fill with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin: Engineer into 'HasCabin' (1 if present, 0 else) and drop original
    # This avoids dropping too many rows or dealing with high cardinality
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df = df.drop(columns=['Cabin'])

    # --- 2. Feature Selection & Engineering ---
    # Drop irrelevant IDs and Names
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Encode Sex (Binary)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Encode Embarked (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

    # --- 3. Split Features (X) and Target (y) ---
    target_col = 'Survived'
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # --- 4. Scaling ---
    # We fit scaler only on X (will handle data leakage in split logic roughly here, 
    # but for simplicity in this utility we scale all before splitting or split then scale.
    # Best practice: Split THEN scale.
    
    # First Split: Train (70%) vs Temp (30%)
    train_size = split_ratio[0]
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    # Second Split: Val (15%) vs Test (15%) -> relative 0.5 split of Temp
    val_relative = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_relative, random_state=random_state, stratify=y_temp
    )

    # Scale using ONLY training data statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val,   'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': df.drop(columns=[target_col]).columns.tolist(),
        'scaler': scaler
    }

def preprocess_iris(data_dir: Path) -> Dict[str, Any]:
    """
    Preprocesses the Iris dataset.
    Returns X (scaled features) and y (encoded species).
    """
    file_path = data_dir / "iris.csv" 
    # Handle case where file might be named differently or from sklearn
    if not file_path.exists():
        # Fallback to sklearn if CSV missing, but strictly trying CSV first
        try:
            from sklearn.datasets import load_iris
            iris = load_iris()
            X = iris.data
            y = iris.target
            feature_names = iris.feature_names
            print("Loaded Iris from sklearn (CSV not found).")
        except:
            raise FileNotFoundError(f"Iris dataset not found at {file_path}")
    else:
        df = pd.read_csv(file_path)
        # Assuming standard format: columns 0-3 are features, last is species
        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
        
        # Encode labels if strings
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        feature_names = df.columns[:-1].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return {
        'X': X_scaled,
        'y': y,
        'feature_names': feature_names,
        'n_clusters': len(np.unique(y))
    }

def preprocess_mall_customers(data_dir: Path) -> Dict[str, Any]:
    """
    Preprocesses the Mall Customers dataset.
    
    Steps:
    1. Drop CustomerID.
    2. Encode Gender.
    3. Scale Age, Income, Score.
    """
    file_path = data_dir / "Mall_Customers.csv"
    if not file_path.exists():
        # Try lowercase
        file_path = data_dir / "mall_customers.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Mall Customers dataset not found in {data_dir}")

    df = pd.read_csv(file_path)

    # Drop non-predictive ID
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Rename confusing columns if necessary for consistency
    df.rename(columns={
        'Annual Income (k$)': 'Income',
        'Spending Score (1-100)': 'Score'
    }, inplace=True)

    # Encode Gender
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])

    # Scale all features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    return {
        'X': X,
        'feature_names': df.columns.tolist(),
        'scaler': scaler
    }