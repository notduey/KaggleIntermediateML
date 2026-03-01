#!usr/bin/env python3
"""
Utility functions
"""
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

def score_model(model,X_train, X_valid, y_train, y_valid):
    """
    Score model MAE on test set
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
