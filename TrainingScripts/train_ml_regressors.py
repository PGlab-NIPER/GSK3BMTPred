#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML Regression Model Training for pIC50 Prediction

Algorithms:
Linear Regression
Ridge
Lasso
KNN
SVR
Random Forest
XGBoost

Training uses:
10-fold cross-validation + GridSearch

Author: Pradnya Kamble
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
CV = KFold(n_splits=10, shuffle=True, random_state=SEED)

# ===============================
# Load Data
# ===============================

X_train = pd.read_csv("X_train_common_features.csv")
X_test = pd.read_csv("X_test_common_features.csv")

y_train = pd.read_csv("y_pic50_train.csv").values.ravel()
y_test = pd.read_csv("y_pic50_test.csv").values.ravel()

# ===============================
# Metric Functions
# ===============================

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def metrics(y_true,y_pred):

    return {
    "R2": r2_score(y_true,y_pred),
    "RMSE": rmse(y_true,y_pred),
    "MAE": mean_absolute_error(y_true,y_pred)
    }

# ===============================
# Models
# ===============================

models = {

"Linear":(LinearRegression(),{}),

"Ridge":(
Ridge(),
{"alpha":[0.1,1,10]}
),

"Lasso":(
Lasso(max_iter=5000),
{"alpha":[0.001,0.01,0.1]}
),

"KNN":(
KNeighborsRegressor(),
{"n_neighbors":[3,5,7],"p":[1,2]}
),

"SVR":(
SVR(),
{"C":[0.1,1,10],"kernel":["rbf","poly"]}
),

"RF":(
RandomForestRegressor(random_state=SEED),
{"n_estimators":[50,100],"max_depth":[10,20]}
),

"XGB":(
XGBRegressor(random_state=SEED),
{"n_estimators":[50,100],"max_depth":[3,4]}
)

}

# ===============================
# Training
# ===============================

results=[]

os.makedirs("models",exist_ok=True)

for name,(model,grid) in models.items():

    print(f"\n===== Training {name} =====")

    if grid:

        gs=GridSearchCV(
        model,
        grid,
        cv=CV,
        scoring="r2",
        n_jobs=-1
        )

        gs.fit(X_train,y_train)

        best_model=gs.best_estimator_

    else:

        best_model=model.fit(X_train,y_train)

    train_pred=best_model.predict(X_train)
    test_pred=best_model.predict(X_test)

    train_metrics=metrics(y_train,train_pred)
    test_metrics=metrics(y_test,test_pred)

    cv_r2=cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=CV,
    scoring="r2"
    )

    results.append({

    "Model":name,

    "CV_R2_Mean":np.mean(cv_r2),
    "CV_R2_STD":np.std(cv_r2),

    "Train_R2":train_metrics["R2"],
    "Test_R2":test_metrics["R2"],

    "Train_RMSE":train_metrics["RMSE"],
    "Test_RMSE":test_metrics["RMSE"],

    "Train_MAE":train_metrics["MAE"],
    "Test_MAE":test_metrics["MAE"]

    })

    joblib.dump(best_model,f"models/{name}_regressor.pkl")

# ===============================
# Save Results
# ===============================

results_df=pd.DataFrame(results)

results_df.to_csv("regression_model_metrics.csv",index=False)

print("\nTraining complete.")
