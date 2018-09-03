import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# here 2=======+++++++
def cross_validate(X, y):
    lr = LogisticRegression()
    dtc = DecisionTreeClassifier()
    # svc = SVC()
    mlp = MLPClassifier()
    gs = GaussianNB()
    rgc = RandomForestClassifier()
    gbc = GradientBoostingClassifier()
    xgbc = XGBClassifier()
    gbm = LGBMClassifier()
    cv1 = cross_val_score(lr, X, y, scoring="roc_auc", cv=5)
    print("逻辑回归：", cv1.mean())
    cv2 = cross_val_score(dtc, X, y, scoring="roc_auc", cv=5)
    print("决策树：", cv2.mean())
    # cv3 = cross_val_score(svc, X, y, scoring="roc_auc", cv=5)
    # print("支持向量机：", cv3.mean())
    cv4 = cross_val_score(mlp, X, y, scoring="roc_auc", cv=5)
    print("神经网络：", cv4.mean())
    cv5 = cross_val_score(gs, X, y, scoring="roc_auc", cv=5)
    print("高斯朴素贝叶斯：", cv5.mean())
    cv6 = cross_val_score(rgc, X, y, scoring="roc_auc", cv=5)
    print("随机森林：", cv6.mean())
    cv7 = cross_val_score(gbc, X, y, scoring="roc_auc", cv=5)
    print("gbc：", cv7.mean())
    cv8 = cross_val_score(xgbc, X, y, scoring="roc_auc", cv=5)
    print("xgbc：", cv8.mean())
    cv9 = cross_val_score(gbm, X, y, scoring="roc_auc", cv=5)
    print("gbm：", cv9.mean())
    # 逻辑回归： 0.6913635296951177
    # 决策树： 0.9356012111869803
    # 神经网络： 0.6667293085576066
    # 高斯朴素贝叶斯： 0.669953487168708
    # 随机森林： 0.9868992014626814
    # gbc： 0.9799233421521952
    # xgbc： 0.9704163964351045
    # gbm： 0.9840698808306522

# def gridSearchCV(X, y):
#     params = {"min_samples_leaf": [1, 3, 5],
#               "n_estimators": [30, 100, 150],
#               "max_depth": [None, 7, 10]}
#     gd = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring="f1")
#     gd.fit(X, y)
#     print(gd.best_score_)
#     print(gd.best_params_)
#     # 0.9677037705489846
#     # {'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 150}
#
# def train(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=1)
#     model = RandomForestClassifier(min_samples_leaf=1, n_estimators=150, max_depth=None)
#     # 50wan数据
#     # 0.9686611385275826
#     #                 precision   recall  f1 - score  support
#     # 0               0.99        0.95    0.97        108990
#     # 1               0.95        0.99    0.97        109046
#     #
#     # avg / total     0.97        0.97    0.97        218036
#     #
#     # [[103222  5768]
#     #  [1065 107981]]
#     model.fit(X_train, y_train)
#     pre_y = model.predict(X_test)
#     print(accuracy_score(y_test, pre_y))
#     print(classification_report(y_test, pre_y))
#     print(confusion_matrix(y_test, pre_y))
#     # 300wan数据
#     # 0.9683855041615761
#     #                 precision   recall  f1 - score  support
#     # 0               0.99        0.95    0.97        838606
#     # 1               0.95        0.99    0.97        837682
#     # avg / total     0.97        0.97    0.97        1676288
#     #
#     # [[793672  44934]
#     #  [8061 829621]]

def gridSearchCV(X, y):
    params = {"num_leaves": [35, 40, 50],
              "n_estimators": [500, 550, 570],
              "learning_rate": [0.01, 0.1, 1]}
    gd = GridSearchCV(LGBMClassifier(), param_grid=params, scoring="f1")
    gd.fit(X, y)
    print(gd.best_score_)
    print(gd.best_params_)
    # 0.9666670682642582
    # {'learning_rate': 0.1, 'n_estimators': 550, 'num_leaves': 40}

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=1)
    model = LGBMClassifier(num_leaves=40, n_estimators=550, learning_rate=0.1)
    # 50wan数据
    # 0.9691014327909152
    #                 precision   recall  f1 - score  support
    # 0               0.99        0.94    0.97        108990
    # 1               0.95        0.99    0.97        109046
    #
    # avg / total     0.97        0.97    0.97        218036
    #
    # [[102845  6145]
    #  [592 108454]]
    model.fit(X_train, y_train)
    pre_y = model.predict(X_test)
    print(accuracy_score(y_test, pre_y))
    print(classification_report(y_test, pre_y))
    print(confusion_matrix(y_test, pre_y))
    fpr, tpr, th = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr)
    plt.title("ROC")
    plt.show()
    # 0.9692302277412951
    #                 precision   recall  f1 - score  support
    # 0               1.00        0.94    0.97        838606
    # 1               0.95        1.00    0.97        837682
    #
    # avg / total     0.97        0.97    0.97        1676288
    #
    # [[790541  48065]
    #  [3514 834168]]

def data_cleaning(df):
    print(df.info())
    df.drop(["SUBS_INSTANC", "IS_SANWU", "MEMBER_LVL", "PRODUCT_TYPE", "AGREE_TYPE", "CALL_DAYS", "VALID_CALL_RING",
             "CELLID_NUM", "YIWANG_CNT"], inplace=True, axis=1)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    cols = ["CHANNEL_TYPE", "INNET_MONTH", "ACCT_FEE", "JF_TIMES", "P2P_SMS_CNT", "ACCT_CN", "TIMES_CN", "CALL_CN7"]
    for i in cols:
        df.drop(df[df[i].isin([99])].index, axis=0)
    # 0:16 = 1:16
    print(df["IS_LOST"][df["IS_LOST"] == 0].count() / df["IS_LOST"][df["IS_LOST"] == 1].count())
    X = df.iloc[:, : -1]
    y = df.iloc[:, -1]
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(X, y)

    return X_smo, y_smo

def main():
    time1 = datetime.datetime.now()
    df = pd.read_csv("lost_data.csv")
    X_smo, y_smo = data_cleaning(df)
    # cross_validate(X_smo, y_smo)
    # gridSearchCV(X_smo, y_smo)
    train(X_smo, y_smo)
    time2 = datetime.datetime.now()
    print(time1)
    print(time2)

if __name__ == '__main__':
    main()