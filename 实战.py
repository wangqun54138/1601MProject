from __future__ import division
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
# new new day
churn_df = pd.read_csv('churn.csv')
col_names = churn_df.columns.tolist()

to_show = col_names[:6] + col_names[-6:]

churn_result = churn_df["Churn?"]

y = np.where(churn_result == 'True.', 1, 0)

to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_feat_space = churn_df.drop(to_drop, axis=1)

# 将这些yes和no转化为布尔值
yes_no_cols = ["Int'l Plan", "VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

feaures = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

def run_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()
    y_tests = y.copy()
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]

        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
        y_tests[test_index] = y[test_index]
    return y_pred, y_tests
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

def accuracy(y_true, y_pred, y_tests, title):
    # cnf_matrix = confusion_matrix(y_tests, y_pred)
    return np.mean(y_true == y_pred)

def run_prob_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y), 2))

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob


pred_prob = run_prob_cv(X, y, RF, n_estimators=10)

pred_churn = pred_prob[:, 1]
is_churn = y == 1

counts = pd.value_counts(pred_churn)

true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

counts = pd.concat([counts, true_prob], axis=1).reset_index()
counts.columns = ["pred_prob", "count", "true_prob"]
print(counts)





