#!usr/bin/env python3
"""
Section 3: Pipelines
"""
from utilities import load_split_melb

# Pipelines are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.
# Many data scientists hack together models without them, but pipelines have some important benefits:
# - Cleaner code: accounting for data at each step of preprocessing can get messy, With a pipeline, you won't need to manually keep track of your training and validation data.
# - Fewer bugs: there are fewer opportunities to misapply a step or forget a preprocessing step.
# - Easier to productionize: It can be surprisingly hard to transition a model from a prototype to something deployable at scale. We won't go deep into the concerns relating to productionization, but a pipeline can help.
# - More options for model validation: With a pipeline, you can use options like cross-validation (next section) to validate your model.

# Load and split data
X_train_full, X_valid_full, y_train, y_valid = load_split_melb()

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [ # list of categorical columns AND have < 10 unique values
    cname for cname in X_train_full.columns # loop through every column
    # 'expression for item in iterable if condition', expression = value that's appended to list
    if X_train_full[cname].nunique() < 10 # checks cardinality, if column has < 10 unique values and
    and X_train_full[cname].dtype == "string" # column is data type 'string' (e.g. text)
]

# Select numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns # loop through every column name
    if X_train_full[cname].dtype in ['int64', 'float64'] # if column datatype is int64 or float64
]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols # list of numerical and cardinality-filtered columns
X_train = X_train_full[my_cols].copy() # copy of X_train_full with only selected columns
X_valid = X_valid_full[my_cols].copy()

print(X_train.head())

# We'll construct the full pipeline in three steps:

# 1. Define preprocessing steps
# Similar to how a pipeline bundles together preprocessing and modeling steps, we use the ColumnTransformer class to bundle together preprocessing steps.
# Code below imputes missing values in numerical data, and imputes missing values and applies a one-hot encoding to categorical data:
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant') # constant imputes with fill_value parameter (default is None, imputes 0 for numeric data and "missing value" for categorical data)

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[ # pipeline is a list of steps to perform
    ('imputer', SimpleImputer(strategy='most_frequent')), # imputes with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # applies one-hot encoding
    # handle_unknown='ignore' results in row with unknown value to have all zeros
    # e.g. training data shows "Red", "Blue", "Green", but test data has "Yellow":
    #   Red  Blue  Green
    #   0    0     0
])
# Pipelines objects define the steps when transforming data
# each step is a tuple ('name_of_step', transformer_object)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer( # routes specific columns to specific transformers
    transformers=[ # tuple list, parameters: (name, transformer_object, columns_to_apply_to)
        ('num', numerical_transformer, numerical_cols), # tranform numerical_col with numerical_transformer
        ('cat', categorical_transformer, categorical_cols) # transform categorical_cols with categorical_transformer pipeline (impute then one-hot encode)
    ]
)

# 2. Define the model
# Now we'll define a random forest model with RandomForestRegressor class
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

# 3. Create and evaluate the pipeline
# Finally we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps. Important things to note:
# - with the pipeline, we preprocess the training data and fit the model in a single line of code. In contrast, without a pipeline, we would have to do imputation, one-hot encoding, and model training in separate steps/lines of code. This becomes especially messy if we have to deal with both numerical and categorical variables!
# - with the pipeline, we supply the unprocessed features in 'X_valid' to the predict() method, and the pipeline automatically preprocesses the features before generating predictions. However, without a pipeline, we have to remember to preprocess the validation data before making predictions.
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # preprocess data
    ('model', model) # define model
])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
# MAE: 160623.3888096991

# Conclusion: pipelines are valuable for cleaning up machine learning code and avoiding errors, and are especially useful for workflows with sophisticated data preprocessing.