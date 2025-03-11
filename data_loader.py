"""
Data loading utilities for the Multiple Linear Regression model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_csv_data(file_path, target_column, feature_columns=None):
    """
    Load data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    target_column : str
        Name of the target variable column
    feature_columns : list, optional
        List of feature column names. If None, uses all columns except target
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Extract features and target
        X = data[feature_columns]
        y = data[target_column]
        
        print(f"Data loaded successfully from {file_path}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Number of samples: {len(data)}")
        
        return X, y
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def load_excel_data(file_path, target_column, feature_columns=None, sheet_name=0):
    """
    Load data from an Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    target_column : str
        Name of the target variable column
    feature_columns : list, optional
        List of feature column names. If None, uses all columns except target
    sheet_name : str or int, optional
        Name or index of the sheet to read
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    """
    try:
        # Read the Excel file
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Extract features and target
        X = data[feature_columns]
        y = data[target_column]
        
        print(f"Data loaded successfully from {file_path}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Number of samples: {len(data)}")
        
        return X, y
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    test_size : float, optional
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Split data
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 