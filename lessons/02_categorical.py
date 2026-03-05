#!usr/bin/env python3
"""
Section 2: Categorical Variables
"""
import pandas as pd
from utilities import load_split_melb, score_model

# Categorical variables are variables that take on a limited number of possible values
# Consider a survey that asks how often you eat breakfast and provides four options: "Never", "Rarely", "Most days", or "Every day." In this case, the data is categorical because responses fall into a fixed set of categories.
# If people reponses to a survey about which brand of car they owned, the reponses would fall into categories like "Honda," "Toyota," "Ford," etc. In this case the data is also categorical.
# You will get an error if you try to plug these variables into most machine learning models in Python without preprocessing them first.

# Three approaches to handle categorical variables:
# 1. Drop categorical variables
# The easiest approach to dealing with categorical variables is to simply remove them from the dataset. This approach will only work well if the columns did not contain useful information.

# 2. Ordinal Encoding
# Ordinal encoding assigns each unique value to a different integer e.g. for the breakfast question, "Never" = 0, "Rarely" = 1, "Most days" = 2, "Every day" = 3.
# This assumption makes sense in this example, because there is an indisputable ranking to the categories. Not all categorical variables have a clear ordering in the values, but we refer to those that do as ordinal variables. For tree-based models (like decision trees and random forests), you can expect ordinal encoding to work well with ordinal variables.

# 3. One-Hot Encoding
# One-hot encoding creates a new column indicating the presence (or adsence) of each possible value in the original data. To understand this, we'll work through an example:
# Color  -->  Red  Yellow  Green
# Red          1     0      0
# Red          1     0      0
# Yellow       0     1      0
# Green        0     0      1
# Yellow       0     1      0
# In the original dataset, "Color" is a categorical variable with three categories, "Red", "Yellow", and "Green". The corresponding one-hot encoding contains one column for each possible value, and one row for each row in the original dataset. Wherever the original value was "Red", we put a 1 in the "Red" column; if the original value was "Yellow", we put a 1 in the "Yellow" column, and so on.
# In contrast to ordinal encoding, one-hot encoding does not assume an ordering of the categories. This, you can expect this approach to work particularly well if there is no clear ordering in the categorical data (e.g. "Red" is neither more or less than "Yellow"). We refer to categorical variables without an intrinsic ranking as nominal variables.
# One-hot encoding generally does not perform well if the categorical variable takes on a large number of values (i.e. you generally won't use it for variables taking more than 15 values).

# Load and split data
X_train_full, X_valid_full, y_train, y_valid = load_split_melb()

# Drop columns with missing value (simplest approach)
cols_with_missing = [ # list of columns missing at least one value
    col for col in X_train_full.columns # loop through every column
    if X_train_full[col].isnull().any() # checks if any values in column are missing
]
X_train_full.drop(cols_with_missing, axis=1, inplace=True) # drop columns with missing values
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [ # list of categorical columns that are store as text and have < 10 unique values
    cname for cname in X_train_full.columns # loop through every column
    if X_train_full[cname].nunique() < 10 # checks cardinality, if column has < 10 unique values and
    and X_train_full[cname].dtype == "string" # column is data type 'string' (e.g. text)
]
# cname for cname in X_train_full.columns -> first cname is what is appended to the list, second cname is loop variable, use same name for both because you want to add the column name itself to the list
# X_train_full[cname] is one column (pandas series) containing all values in that column
# .unique() < n checks if there are fewer than n unique values in that column
# .dtype == ___ checks column datatype

# Select numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns # loop through every column name
    if X_train_full[cname].dtype in ['int64', 'float64'] # if column datatype is int64 or float64
]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols # list of numerical and cardinality-filtered columns
X_train = X_train_full[my_cols].copy() # copy of X_train_full with only selected columns
X_valid = X_valid_full[my_cols].copy()

print(X_train.head())

# next we obtain a list of all of the categorical variables in the training data
# we do this by checking the data type (or dtype) of each column. The object (now string) dtype indicates a column has text (there are other things it could theoretically be, but that's unimportant for our purposes). For this dataset, the columns with text indicate categorical variables.

# Get list of categorical variables
s = (X_train.dtypes == 'string') # pandas series of booleans relating each column if dtype is string
object_cols = list(s[s].index) # list of string dtype columns, s[s] is boolean mask (keep only True columns)

print("Categorical variables:")
print(object_cols)

from sklearn.ensemble import RandomForestRegressor

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Score from approach 1 (drop categorical variables)
drop_X_train = X_train.select_dtypes(exclude=['object']) # exclude categorical variables (object)
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_model(model, drop_X_train, drop_X_valid, y_train, y_valid))
# MAE from Approach 1 (Drop categorical variables): 
# 175707.61156991488

# Score from approach 2 (ordinal encoding)
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder() # create ordinal encoder
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols]) # fit and transform
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols]) # transform valid data
# we don't fit the ordinal encoder to the validation data because we want to use the same encoding/category mapping as the training data, fitting on validation data might get a different mapping

print("MAE from Approach 2 (Ordinal Encoding):")
print(score_model(model, label_X_train, label_X_valid, y_train, y_valid))
# MAE from Approach 2 (Ordinal Encoding): 
# 165919.14549617787

# For the encoder above, we randomly assign each unique value to a different integer. This is a common apprach that is simpler than providing custom labels; however we can expect an additional boost in performance if we provide better-informed labels for all ordinal variables.

# Score from approach 3 (one-hot encoding)
# We use the OneHotEncoder class. There are a number of parameters that can be used to customize its behavior:
# -we'll set handle_unknown='ignore' to avoid errors when validation data contains classes that aren't seen in the training data.
# -setting sparse_output=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols])) # fit and transform
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols])) # transform valid data

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1) # combine numerical and one-hot encoded columns
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all one-hot encoded columns have type string
OH_X_train.columns = OH_X_train.columns.astype(str) # convert columns names to strings
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):")
print(score_model(model, OH_X_train, OH_X_valid, y_train, y_valid))
# MAE from Approach 3 (One-Hot Encoding):
# 166111.84115541063

# Which approach is best?
# In this case, dropping the categorical columns (approach 1) performed worst, since it had the highest MAE score (175707.61156991488). As for the other two approaches, since the return MAE scores are so close in value, there doesn't appear to be any meaningful benefit to one over the other(165919.14549617787 vs 166111.84115541063).
# In general, one-hot encoding (approach 3) will typically perform best, and dropping the categorical columns (approach 1) typically perform worst, but it varies on a case-by-case basis.