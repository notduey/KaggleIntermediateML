#!/usr/bin/env python3
"""
Section 6: Data Leakage
"""
import pandas as pd

# Data leakage happens when your training data contains information about the target, but similar data will not be available when the mode is used for predictions. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly on new data or in the real world.
# In other words, leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.
# There are two main types of leakage: target leakage and train-test contamination.

# Target leakage occurs when your predictors include data that will not be available at the time your make predictions. It is important to think about target leakage in terms of the timing or chronological order that data becomes available, not merely whether a feature helps make good predictions.
# Imagine you want to predict who will get sick with pneumonia. The top few rows of your raw data look like this:

# got_pneumonia | age | weight |  male  | took_antibiotic_medicine | ...
# --------------+-----+--------+--------+--------------------------+-----
# False         | 65  |   100  |  False |           False          | ...
# False         | 72  |   130  |  True  |           False          | ...
# True          | 58  |   100  |  False |           True           | ...

# People take antibiotic medicines AFTER getting pneumonia in order to recover. The raw data shows a strong relationship between those columns, but 'took_antibiotic_medicine' is frequently changed after the value for 'got_pneumonia' is determined. This is target leakage.
# The model would see that anyone who has a value of 'False' for 'took_antibiotic_medicine' didn't have pneumonia. Since validation data comes from the same source as training data, the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores.
# But the model will be very inaccurate when subsequently deployed in the real world, because even patients who will get pneumonia won't have received antibiotics yet when we need to make predictions about their future health.
# To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.

# Train-test contamination is a different type of leakage that occurs when you aren't careful to distinguish training data from validation data.
# Recall that validation is meant to be a measure of how the model does on data that it hasn't considered before. You can corrupt this process in subtle ways if the validation data affects the preproessing behavior. This is sometimes called train-test contamination.
# For example, imagine you run preprocessing (like fitting an imputer for missing values) before calling 'train_test_split()'. The end result? Your model may get good validation scores, giving you great confidence in it, but perform poorly when you deplooy it to make decisions.
# After all, you incorporated data from the validation or test data into how you make predictions so it may do well on that particular data even if it can't generalize to new data. This problem becomes even more subtle (and more dangerous) when you do more complex feature engineering.
# If your validaiton is based on a simple train-test-split, exlude the validation data from any type of fitting, including the fitting of preprocessing steps. This is easier if you use scikit-learn pipelines. When using cross validation, it's even more critical that you do your preprocessing inside the pipeline!

# In this example, we'll use one method to detect and remove target leakage. We will use a dataset about credit card applications. The end results is that information about each credit card application is stored in a DataFrame 'X'. We'll use it to predict which applications were accepted in a Series 'y'.

# Read the data
data = pd.read_csv('data/AER_credit_card_data.csv', true_values=['yes'], false_values=['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0]) # 1319
print(X.head())

# Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best pratice)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, cv=5, scoring='accuracy')

print(f"Cross-val accuracy: {cv_scores.mean()}") # 0.980294s

# With experience, you'll find that it's very rare to find models that are accurate 98% of the time. It happens, but it's uncommon enough that we should inspect the data more closely for target leakage. Here is a summary of the data:

# card: 1 if credit card application accepted, 0 if not
# reports: Number of major derogatory reports
# age: Age n years plus twelfths of a year
# income: Yearly income (divided by 10,000)
# share: Ratio of monthly credit card expenditure to yearly income
# expenditure: Average monthly credit card expenditure
# owner: 1 if owns home, 0 if rents
# selfempl: 1 if self-employed, 0 if not
# dependents: 1 + number of dependents
# months: Months living at current address
# majorcards: Number of major credit cards held
# active: Number of active credit accounts

# A few variables look suspicious. For example, does 'expenditure' mean ependiture on this card or on cards used before applying?
# At this point, basic data comparisons can be very helpful

expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print(f'Fraction of those who did not receive a card and had no expenditures: \
      {((expenditures_noncardholders == 0).mean()):.2f}') # 1.00

print(f'Fraction of those who received a card and had no expenditures: \
      {((expenditures_cardholders == 0).mean()):.2f}') # 0.02

# As shown above, everyone who did not receieve a card has no expenditures, while only 2% of those who received a card has expenditures. It's not surprising that our model appeared to have high accuracy. But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.
# Since 'share' is partially determined by 'expenditure', it should be excluded too. The variables 'active' and 'majorcards' are a little less clear, but from the decripstion, they sound concerning. In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.

# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, cv=5, scoring='accuracy')

print(f"Cross-val accuracy: {cv_scores.mean()}") # 0.8256135499481507

# The accuracy is quite a bit lower, which might be dissapointing. However, we can expect it to be right about 80% of the time when used on new applications, whereas the leaky model woulf likely do much worse than that (in spite of its higher apparent socre in cross-validation).

# Conclusion:
# Data leakage can be a multi-million dollar mistake in many data science applications. Care separation of training and validation data can prevent train-test contamination, and pipelines can help implement this separation. Likewise, a combination of caution, common sense, and data exploration can help identify target leakage.