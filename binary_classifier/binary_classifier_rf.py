import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

file1 = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_28/final_28_2%_OUT.CSV'

lb = LabelBinarizer()

def start():
    dataset = pd.read_csv(file1)

    X = dataset.iloc[:,1:22]
    y = dataset.iloc[:,22:23]

    X, y = SMOTE().fit_resample(X, y)
    data = pd.concat([X,y], axis=1)

    for x in range(3):
        data_standard = data.loc[data['target'] == x]
        data_compare = data[data['target'] != x].sample(frac=0.5, random_state=1)
        data_standard['target'] = 0
        data_compare['target'] = 1

        data_new = pd.concat([data_standard,data_compare], axis=0)
        X_new = data_new.iloc[:,0:21]
        y_new = data_new.iloc[:,21:22]
        process2(X_new,y_new)

def process2(X, y):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    y_train_lb = lb.fit_transform(y_train)
    y_validation_lb = lb.fit_transform(y_validation)

    #model = xgb_hyperparameter(XGBClassifier())
    #model = rf_hyperparameter(RandomForestClassifier())
    model = RandomForestClassifier()
    model = model.fit(X_train, y_train_lb)

    predictions = model.predict(X_validation)

    print(classification_report(y_validation_lb,predictions))

    return model

def rf_hyperparameter(model):
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=5, 
    scoring='roc_auc', n_jobs=4, cv=3, verbose=3)

    return random_search

def xgb_hyperparameter(model):
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=5, 
    scoring='roc_auc', n_jobs=4, cv=3, verbose=3)

    return random_search

if __name__ == '__main__':
    model = start()
