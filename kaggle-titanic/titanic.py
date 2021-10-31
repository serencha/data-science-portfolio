"""
Created by Serena Chang

"""


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
Summary:
    Use logistic regression to predict survivability on
    the Titanic. Structure:
        1. Remove outliers
        2. Check for correlation
        3. Make dummy variables
        4. Check for collinearity
        5. Make a Logistic Regression model
        6. Export predictions

Results:
    Accuracy: 0.7758
    Precision: 0.7034
    Recall: 0.7364
    Score on Kaggle: 0.73444
"""

# Instantiate data, clean and convert Categorical variables from text to binary
train = pd.read_csv('kaggle-titanic/train.csv')

# Remove outliers for numerical columns using SD method


def remove_outliers(df):
    rows_to_remove = []
    for col in df:
        if df[col].dtype.kind in 'biufc':
            mean, std = np.mean(df[col]), np.std(df[col])
            cut_off = std * 3
            lower, upper = mean - cut_off, mean + cut_off
            outliers = [x for x in range(
                len(df[col])) if df[col][x] < lower or df[col][x] > upper]
            #print('Identified outliers: %d' % len(outliers))

            outliers_removed = [x for x in range(
                len(df[col])) if df[col][x] >= lower and df[col][x] <= upper]
            #print('Non-outlier observations: %d' % len(outliers_removed))
            rows_to_remove = rows_to_remove + outliers
    df = df.drop(rows_to_remove)
    return df


df = remove_outliers(train)
# print(len(df))


# Convert columns to dummy variables
df['Sex_'] = np.select(
    [df['Sex'] == 'male', df['Sex'] == 'female'], [0, 1])
df['Cabin_'] = np.select(
    [df['Cabin'] == 'None', df['Cabin'] != 'None'], [0, 1])
df['Embarked_'] = np.select(
    [df['Embarked'] == 'S', df['Embarked'] == 'C', df['Embarked'] == 'Q'], [0, 1, 2])
df['Fare'].fillna(value=df.Fare.mean(), inplace=True)

# Remove all leftover null values
#df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# Check for correlation
cols = ['Pclass', 'Sex_', 'Age', 'SibSp',
        'Parch', 'Fare', 'Cabin_', 'Embarked_']
y = df['Survived']


def find_corr(cols, y):
    list_var = []
    for col in cols:
        X = df[col]
        r = X.corr(y)
        if abs(r) >= .30:
            list_var.append(col)
        print(f"{col} Correlation: {r}")
    return list_var


list_var = find_corr(cols, y)
print(list_var)
X_var = df[list_var]


# Check for multicollinearity
vif = pd.DataFrame()
vif["variable"] = X_var.columns
vif["vif"] = [variance_inflation_factor(
    X_var.values, i) for i in range(len(X_var.columns))]
print(vif)


# Split data into train and test
col_names = ['Pclass', 'Sex_B', 'Age', 'Company',
             'Fare', 'Cabin_B', 'Embarked_B']
X = df[list_var]
y = df.Survived
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=2)


# Instantiate the model
# Fit the model with data and predict survival
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Evaluate performance
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred).round(4))
print("Precision: ", metrics.precision_score(y_test, y_pred).round(4))
print("Recall: ", metrics.recall_score(y_test, y_pred).round(4))

# Predict results for test.csv & export predictions
test = pd.read_csv('kaggle-titanic/test.csv')
test['Sex_'] = np.select(
    [test['Sex'] == 'male', test['Sex'] == 'female'], [0, 1])
test['Fare'].fillna(value=test.Fare.mean(), inplace=True)
X = test[list_var]
print(X)
test['Survived'] = log_reg.predict(X)
print(test)
print(test.columns)
final = test[['PassengerId', 'Survived']]
final.to_csv('submission.csv', index=False)
