#!usr/bin/env python3
"""
Section 4: Cross-Validation
"""
import pandas as pd

# In machine learning, you will face choices about what predictive variables to use, what types of models to use, what arguments to supply to those models, etc. Measuring model quality with a validation (holdout) set is a common practice.
# There are drawbacks to this approach. Imagine a dataset with 5000 rows. You will typically keep about 20% of the data as a validation dataset, or 1000 rows in this case. This leaves some random chance in determining model scores. That is, a model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows.
# At an extreme, you could imagine having only 1 row of data in the validation set. If you compare alerntive models, whihch one makes the best predictions on a single data point will be mostly a matter of luck.
# In general, the large the validation set, the less randon (noise) there is in our evaluation of model quality, and the more reliable it will be. Unfortunately, we can only get a large validation set by removing rows from our training data, and small training datasets mean worse models!

# This is where Cross-Validation comes in
# In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.
# For example, we could begin by divding the data into 5 pieces, each 20% of the full dataset. In this case, we say that we have broken the data into 5 "folds".

# Example of 5-fold cross-validation:
# 1st fold   |   2nd fold   |   3rd fold   |   4th fold   |   5th fold
#------------+--------------+--------------+--------------+-------------
# VALIDATION |   training   |   training   |   training   |   training
# training   |   VALIDATION |   training   |   training   |   training
# training   |   training   |   VALIDATION |   training   |   training
# training   |   training   |   training   |   VALIDATION |   training
# training   |   training   |   training   |   training   |   VALIDATION

# In the first fold, we use the data in the first fold as the validation set, and the other 4 fold as the training set.
# In the second fold, we use the data in the second as validation, and the other folds as training sets, and we repeat this process using every fold as the holdout set.
# This way, 100% of the data is used as validation at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if not all are used simultaneously).

# When to use Cross-Validation?
# Cross-validation gives a more accurate measure of model quality, which is important if you are making a lot of modeling decisions. However, it can take longer to run, because it estimates multiple models (one for each fold).
# For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.
# For large datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.
# There's no concrete threshold for what constitutes a large vs. small dataset, but if your model takes a couple minutes or less to run, it's probably worth switching to cross validation.
# Alternatively, you can run cross-validation and see if the scores for each experiment seem close. If each experiment yields the same results, a single validation set is probably sufficient.

# Read the data
data = pd.read_csv('data/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# We define a pipeline that uses an imputer to fill in missing values and a random forest model to make predictions.
# While it's possible to do cross-validation without pipelines, it is more difficult. Using a pipeline will ,ake the code remarkably straightforward.
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# We obtrain the cross-validation scores with the cross_val_score() function from scikit-learn. We set the number of folds with the 'cv' parameter.
from sklearn.model_selection import cross_val_score

# Multiply by -1 since scikit-learn calculates *negative* MAE
scores = -1 * cross_val_score( # split data, fit model, compute score for all folds
    my_pipeline, # pipeline used to fit the data
    X, y, # variables and target values
    cv=5, # number of folds, in this case 5
    scoring='neg_mean_absolute_error'
    # neg_mean_absolute_error transforms the MAE into a negative value, so that higher scores are better, e.g. -5 is higher than -10
)

print("MAE scores:\n", scores)
# [301615.03450966 303143.49091344 287316.47967778 236087.57655869 260385.2736474 ]

# The scoring parameter chooses a measuer of model quality to report: in this case, we chose negative mean absolute error (MAE). The scoring docs for sklearn can be found here: https://scikit-learn.org/stable/modules/model_evaluation.html
# Scikit-learn's scoring convention is that all metrics are defined so a high number is better. Using negatives allow them to be consistent with that convention (negative MAE is unheard of elsewhere).

# We typically want a single measure of model quality to compare alternative models, so we take the average across experiments.
print("Average MAE score (across experiments):\n", scores.mean())


# Conclusion:
# Using cross-validation yields a much better measure of model quality, with the added benefit of cleaning up our code.
# Note that we no longer need to keep track of separate training and validation sets. So, especially for small datasets, it's a good improvement.
