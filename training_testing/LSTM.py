import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint  
from random import choice
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from skmultilearn.problem_transform import BinaryRelevance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from keras.utils import to_categorical
from os import chdir
from glob import glob
#final_28_2%_OUT
file1 = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_28/final_28_2%_OUT.CSV'

lb = LabelBinarizer()
under = RandomUnderSampler()

def start():
    dataset = pd.read_csv(file1)

    '''array = dataset.values
    X = array[:,1:22]
    y = array[:,22:23]'''
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

    #model = LGBMClassifier()
    #model = LGBM_hyperparameter(LGBMClassifier())
    #model = XGBClassifier()
    model = xgb_hyperparameter(XGBClassifier())
    model = model.fit(X_train, y_train_lb)

    predictions = model.predict(X_validation)

    print(classification_report(y_validation_lb,predictions))

    return model

def process():
    dataset = pd.read_csv(file1)
    
    array = dataset.values
    X = array[:,1:22]
    y = array[:,22:23]

    X, y = SMOTE().fit_resample(X, y)

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    y_train_categorical = to_categorical(y_train,3)
    y_validation_categorical = to_categorical(y_validation,3)

    y_train_lb = lb.fit_transform(y_train)
    y_validation_lb = lb.fit_transform(y_validation)

    #model = LGBMClassifier()
    #model = LGBM_hyperparameter(LGBMClassifier())
    #model = XGBClassifier()
    model = RandomForestClassifier()
    #model = rf_hyperparameter(RandomForestClassifier())
    model = model.fit(X_train, y_train)
    #print(model.best_estimator_)

    predictions = model.predict(X_validation)

    print(classification_report(y_validation,predictions))

    return model

def LGBM_hyperparameter(model):
    param_dist = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
              'num_leaves': sp_randint(6, 50), 
              'min_child_samples': sp_randint(100, 500), 
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc = 0.2, scale = 0.8), 
              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
              'colsample_bytree': sp_uniform(loc = 0.4, scale = 0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=2, n_jobs = -1)

    return random_search

def catboost_hyperparameter(model, X_train, y_train):
    parameters = {'depth'         : [4,5,6,7,8,9, 10],
                 'learning_rate' : [0.01,0.02,0.03,0.04],
                  'iterations'    : [10, 20,30,40,50,60,70,80,90, 100]
                 }
    
    random_search = RandomizedSearchCV(estimator = model, param_distributions = parameters, cv = 2, n_jobs=-1)

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
    n_jobs=4, cv=3, verbose=3)

    return random_search

def rf_hyperparameter(model):
    param_grid = {'n_estimators': [1800, 2000],
            'max_features': ['sqrt'],
            'max_depth': [20, 30],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'bootstrap': [True, False]}

    random_search = RandomizedSearchCV(estimator = model, param_distributions = param_grid,
    cv = 3, verbose=0, n_jobs = -1)

    return random_search

def graph_comparison(location_target, location_price, model):
    location_target_1 = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_26/final_26_OUT.CSV'
    location_price_1 = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_25/final_25_OUT.CSV'

    location_target_2 = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_22/final_22_OUT.CSV'
    location_price_2 = '/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_23/final_23_OUT.CSV'
    comparison = pd.read_csv(location_price)
    comparison_array = comparison.values

    dataset = pd.read_csv(location_target)
    dataset['compare'] = comparison_array[:,1:2]
    dataset['p_change'] = comparison_array[:,2:3]
    dataset = dataset.sample(frac=0.2, random_state=1)

    array = dataset.values
    x = array[:,1:22]
    compare = array[:,23:24]
    p_change = array[:,24:25]

    predictions = model.predict(x)
    predictions = lb.inverse_transform(predictions)

    prob1 = model.predict_proba(x)[:,1]

    counter = 0
    total = 0
    x_plot = []
    y_plot = []

    for index, value in enumerate(prob1):
        if value > 0.8:
            total+=1
            x_plot.append(value)
            y_plot.append(p_change[index])
            if compare[index] == 1:
                counter+=1

    ratio = counter/total
    print('Percentage of > 0.8 prob that results in p30-p20 positive change\n' + str(ratio) + '\n')

    plt.figure(figsize=(15,7))
    plt.scatter(x_plot,y_plot,s=10)
    plt.xlabel('Probability')
    plt.ylabel('Ratio change')
    plt.show()

    plt.figure(figsize=(15,7))
    plt.hist(prob1, bins=50, label='Prob of 1', alpha=0.7, color='r')
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('Number of instances', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.show() 

    print('Ratio of predicted 1s to total given values\n' + str(np.count_nonzero(predictions==1) / len(predictions)))

if __name__ == '__main__':
    model = process()

#graph_comparison(location_target_1, location_price_1, model)
#graph_comparison(location_target_2, location_price_2, model)