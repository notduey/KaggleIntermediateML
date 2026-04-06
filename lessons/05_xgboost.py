#!usr/bin/env python 3
"""
Section 5: XGBoost
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Gradient boosting is a popular machine learning optimization technique. This method dominates many Kaggle competitions and achieves state-of-the-art results on a variety of datasets.
# Previously, the random forest model was used, which achieves better performance than a single decision tree simply by averaging the predictions of many decision trees.
# We refer to the random forest as an "ensemble method". By definition, emsemble methods combine the predictions of several models (e.g. several trees, in the case of random forests).

# Gradient Boost:
# Gradiant boosting is a method that goes through cycles to iteratively add models into an ensemble.
# It begins by initalizing the ensemble with a single model, whose predictions can be pretty naive (Even if it's predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors).

# We start with the cycle:
# 1. We use the current ensemble to generate predictions for each observation in the dataset. To make predictions, we add the predictions from all models in the ensemble. These predictions are used to calculate a loss function (like MAE or mean squared error, for instance).
# 2. Then we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model paramters so that adding this new model to the ensembles will reduce loss.
# As a side note: the "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters of the new model.
# 3. Finally we add the new model to the ensemble and repeat the cycle!

# naive model --> make predictions --> calculate loss --> fit new model --> add model to ensemble
#                         ^                                                           |
#                         |___________________________repeat__________________________|

from sklearn.model_selection import train_test_split

# Read data
data = pd.read_csv('data/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# The XGBoost library will be used. XGBoost stands for eXtreme Gradient Boosting, which is an implamentation of gradient boosting with several additional features focused on performance and speed (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages).
# We import the scikit-learn API for XGBoost (xgboost.XGBRegressor). This allows us to build and fit a modle just as we would in scikit-learn.
from xgboost import XGBRegressor

my_model = XGBRegressor()

# Score and evalute model
from utilities import score_model

predictions = score_model(my_model, X_train, X_valid, y_train, y_valid)
print(f"Mean Absolute Error: {str(predictions)}") # 239035.55752025038

# Parameter tuning
# XGBoost has a few parameters that can dramatically affect accuracy and training speed. The first parameters to understand are:

# n_estimators: this specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble. 
# Too low a value causes underfitting, leading to inacurate predictions on both training and test data. Too high a value causes overfitting, leading to accurate predictions on training data, but inaccurate predictions on test data.
# Typical values range from 100-1000, though this depends a lot on the 'learning_rate' parameter below.
my_model = XGBRegressor(n_estimators=500)

# early_stopping_rounds: this offers a way to automatically find the ideal value for 'n_estimators'. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.
# Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. Setting 'early_stopping_rounds=5' is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.
# When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the 'eval_set' parameter. We can modify the above to include early stopping.
my_model = XGBRegressor(
    n_estimators=500,
    early_stopping_rounds=5, # early stopping, stop when score deteriorates for 5 rounds
    )

my_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)], # evaluates validation set after every round and uses that score to decide whether training is improving or not
    verbose=False # false doesn't print output every round
    )

# If you later want to fit a model with all of your data, set 'n_estimators' to whatever value you found to be optimal when run with early stopping.

# learning_rate: instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate). 
# This means each tree we add to the ensemble helps us less. So, we can set a higher value for 'n_estimators' without overfitting. If we use early stopping, the appropriate number of trees will be found automatically.
# In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model long to train since it does more iterations throug the cycle. By default, XGBoost sets 'learning_rate=0.1'.
# Modifying the example above to change the learning rate yields the following code:
my_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=5, # early stopping, stop when score deteriorates for 5 rounds
    )

my_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)], # evaluates validation set after every round and uses that score to decide whether training is improving or not
    verbose=False # false doesn't print output every round
    )

# n_jobs: on larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter 'n_jobs=-1' to use all available cores, or set it equal to the number of cores on your machine. On small datasets, this isn't necessary.
# The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing to worry about. But, it's useful in large datasets where you would otherwise spend a long time waiting during the fitting process.
my_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=-1,
    early_stopping_rounds=5, # early stopping, stop when score deteriorates for 5 rounds
    )

my_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)], # evaluates validation set after every round and uses that score to decide whether training is improving or not
    verbose=False # false doesn't print output every round
    )

# Conclusion:
#XGBoost is a leading software library for working with standard tabular data (the type of data you store in Pandas DataFrame, as opposed to more exotic types of data like images and videos). With careful parameter tuning, you can train highly accurate models.