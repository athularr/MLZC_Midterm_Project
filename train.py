#!/usr/bin/env python
# coding: utf-8

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split
import pickle

output_file = 'engine_xgb_model.bin'

df=pd.read_csv(r"C:\Users\athul\Documents\ML Zoomcamp 2023\Project\Midterm Project\engine_data.csv")

def dataframe_prep(dfx):
    
    dfx.columns=dfx.columns.str.replace(' ', '_').str.lower()
    col_names=list(df.columns)
    dfx_fulltrain, dfx_test=train_test_split(dfx, test_size=0.2, random_state=1)
    dfx_train, dfx_val=train_test_split(dfx_fulltrain, test_size=0.25, random_state=1)
    
    dfx_fulltrain=dfx_fulltrain.reset_index(drop=True)
    dfx_train=dfx_train.reset_index(drop=True)
    dfx_val=dfx_val.reset_index(drop=True)
    dfx_test=dfx_test.reset_index(drop=True)
    
    return dfx_fulltrain, dfx_train, dfx_val, dfx_test, col_names


df1_fulltrain, df1_train, df1_val, df1_test, ss=dataframe_prep(df)


# # Data Preparation

#XGBoost

ss.remove('engine_condition')

dv=DictVectorizer(sparse=False)
fulltrain_dicts=df1_fulltrain[ss].to_dict(orient='records')
test_dicts=df1_test[ss].to_dict(orient='records')
dv.fit(fulltrain_dicts)

#Preparing Training Data
X_full_train= dv.transform(fulltrain_dicts)
y_full_train=df1_fulltrain.engine_condition.values

#Preparing Validation Data
X_test_xg= dv.transform(test_dicts)
y_test_xg=df1_test.engine_condition.values

dfulltrain=xgb.DMatrix(X_full_train, label=y_full_train, feature_names=ss)
dtest=xgb.DMatrix(X_test_xg, label=y_test_xg, feature_names=ss)

watchlist=[(dfulltrain, 'train'), (dtest, 'test')]

max_eta=0.01
max_d=6
max_mcw=10

xgb_params={'eta':max_eta, 
            'max_depth':max_d, 
            'min_child_weight':max_mcw,
            
            'objective':'binary:logistic',
            'eval_metric':'auc',
            
            'nthread': 8, 
            'seed': 1, 
            'verbosity': 1}


xgb_model=xgb.train(xgb_params, dfulltrain, num_boost_round=500, verbose_eval=10, evals=watchlist)

y_pred_xgb=xgb_model.predict(dtest)
y_pred_xgb[:10]


roc_auc_score(y_test_xg, y_pred_xgb)

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, xgb_model, df1_test, y_test_xg, ss), f_out)