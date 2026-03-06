#!usr/bin/env python3
"""
Utility functions
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error

def load_data(path, target_column):
    """
    Load data from csv file
    """
    data = pd.read_csv(path)
    y = data[target_column]
    X = data.drop([target_column], axis=1)
    return X, y

def read_filter_data():
    """
    Read data from train.csv and test.csv and filter out rows with missing target
    """
    root = Path(__file__).resolve().parents[1] # this file's absolute path, then go up 2 directories

    X_full = pd.read_csv(root / 'data/home_data/train.csv', index_col='Id')
    X_test_full = pd.read_csv(root / 'data/home_data/test.csv', index_col='Id')

    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    return X_full, X_test_full, y

def score_model(model, X_train, X_valid, y_train, y_valid):

    """
    Score model MAE on test set
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
