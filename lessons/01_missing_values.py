#!/usr/bin/env python3
"""
Section 1: Missing Values
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

sys.path.append(str(Path(__file__).resolve().parents[1])) # from this file, go up 1 directory

from src.utilities import score_model

# There are many ways data can end up with missing values for example:
# A 2 bedroom house won't include a value for the size of a 3rd bedroom
# A survey respondent may choose not to share their income

# There are 3 approaches to handle missing values:
# 1. Drop columns with missing values
# Unless many values in the dropped column are missing, the model loses a lot of potential information
# An extreme example would be if a dataset with 10,000 rows had a super important column missing even a single entry, this approach would drop the column entirely

# 2. Imputation
# Imputation fills in the missing values with some number
# The most common imputation method is to replace missing values with the mean of the column
# While the inputed value is not perfect, in most cases it usually leads to more accurate models than you would get from dropping an entire column

# 3. Imputation + missing indicator
# Althought imputation is the most common approach, some times the imputed values may be systematically above or below the actual value
# Rows with missing values could also be unique in some way
# Missing indicator columns can help your model make better predications by considering whether a value is missing
# For each column with missing values, add a new column that conveys whether the value is missing
# In some cases, this will improve model performance, but in other cases it doesn't help at all

# Load data
data = pd.read_csv('data/melb_data.csv')

# Select target
y = data.Price

# To keep things simple we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object']) # exclude categorical variables (object)

# Split data into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Define model
model = RandomForestRegressor(n_estimators=10, random_state=0) # n_estimators is number of trees in forest

# Approach 1: drop columns with missing values
cols_with_missing = [ # list of columns with missing values
    col for col in X_train.columns # append col in list
    if X_train[col].isnull().any() # if there is at least one missing value
]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_model(model, reduced_X_train, reduced_X_valid, y_train, y_valid)) # 183550.22137772635

# Approach 2: imputation
my_imputer = SimpleImputer() # by default, imputes mean
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
# fit_transform imputes the means, fit computes the means for each column, transform imputes the means (replaces missing values with the stored means), pd.DataFrame converts to a dataframe
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
# don't fit on validation data because it would cause data leakage, meaning the model would see the distribution of the "future" data and impute different values
# this could mean the model would artificially do better on the validation set, but that ruins the point of validation data if the model already knows the future.

# Imputation removes column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_model(model, imputed_X_train, imputed_X_valid, y_train, y_valid)) # 178166.46269899711
# Filling in the mean value generally performs better but this varies by dataset. There are also other more complex imputation methods (such as regression imputation), however complex strategies typically give little to no additional benefit once you plug the results into sophisticated ML models

# Approach 3: imputation + missing indicator

X_train_plus = X_train.copy() # make copy to void changing original data
X_valid_plus = X_valid.copy()

# For each column in cols_with_missing, make new boolean column indicating whether each row is missing a value
for col in cols_with_missing:
    # make new boolean indicator column and setting to each row whether that row is missing a value
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull() # isnull returns True if missing
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removes column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (Imputation + Missing Indicator):")
print(score_model(model, imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
# 178927.503183954

# Why did approach 2 perform better than approach 1? This is because the training data has 10864 rows and 12 columns, where 3 columns contain missing data. For each column, less than half of the entries are missing. That means dropping the columns removes a lot of useful information for the model, which is why imputation performed better.

# Shape of training data
print(X_train.shape) # returns number of rows and columns

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum()) # sum of true isnull values in each column
print(missing_val_count_by_column[missing_val_count_by_column > 0]) # hide columns with no missing values

# (10864, 12)
# Car               49
# BuildingArea    5156
# YearBuilt       4307
# dtype: int64