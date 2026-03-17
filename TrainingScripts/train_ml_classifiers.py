#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML Classification Model Training for GSK3β Inhibitor Prediction

Algorithms:
KNN
Logistic Regression
Random Forest
XGBoost
SVM

Training uses:
10-fold cross-validation + GridSearch

Author: Pradnya Kamble
"""

# ===============================
# Imports
# ===============================

import os
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ===============================
# Parameters
# ===============================

SEED = 42
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# ===============================
# Load Data
# ===============================

X_train = pd.read_csv("X_train_common_features.csv")
X_test = pd.read_csv("X_test_common_features.csv")

y_train = pd.read_csv("Y_train.csv").values.ravel()
y_test = pd.read_csv("Y_test.csv").values.ravel()

print("Training samples:", X_train.shape)
print("Test samples:", X_test.shape)

# ===============================
# Model Definitions
# ===============================

models = {

"KNN": (
KNeighborsClassifier(),
{
"n_neighbors":[3,5,7],
"p":[1,2]
}
),

"LR": (
LogisticRegression(),
{
"C":[1.0,0.1,0.01],
"solver":["lbfgs","liblinear"],
"penalty":["l1","l2"],
"max_iter":[100,200,300]
}
),

"RF": (
RandomForestClassifier(random_state=SEED),
{
"n_estimators":[50,100,150],
"max_depth":[10,20,30],
"max_features":[5,10,15]
}
),

"XGB": (
XGBClassifier(random_state=SEED),
{
"gamma":[0.5,1,1.5],
"subsample":[0.6,0.8,1.0],
"max_depth":[3,4,5]
}
),

"SVM": (
SVC(probability=True),
{
"C":[0.1,0.5,1],
"gamma":[0.001,0.01,0.1],
"kernel":["linear","rbf","poly"]
}
)

}

# ===============================
# Metric Function
# ===============================

def evaluate(y_true, y_pred, y_prob):

    return {
    "Accuracy": accuracy_score(y_true,y_pred),
    "Precision": precision_score(y_true,y_pred),
    "Recall": recall_score(y_true,y_pred),
    "F1": f1_score(y_true,y_pred),
    "AUC": roc_auc_score(y_true,y_prob)
    }

# ===============================
# Training Loop
# ===============================

results = []

os.makedirs("models", exist_ok=True)

for name,(model,grid) in models.items():

    print(f"\n===== Training {name} =====")

    gs = GridSearchCV(
    model,
    grid,
    cv=CV,
    scoring="accuracy",
    n_jobs=-1
    )

    gs.fit(X_train,y_train)

    best_model = gs.best_estimator_

    # predictions
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    train_prob = best_model.predict_proba(X_train)[:,1]
    test_prob = best_model.predict_proba(X_test)[:,1]

    train_metrics = evaluate(y_train,train_pred,train_prob)
    test_metrics = evaluate(y_test,test_pred,test_prob)

    cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=CV,
    scoring="accuracy"
    )

    results.append({

    "Model":name,
    "Best_Params":gs.best_params_,

    "CV_Accuracy_Mean":np.mean(cv_scores),
    "CV_Accuracy_STD":np.std(cv_scores),

    "Train_Accuracy":train_metrics["Accuracy"],
    "Test_Accuracy":test_metrics["Accuracy"],
    "Test_AUC":test_metrics["AUC"]

    })

    joblib.dump(best_model,f"models/{name}_classifier.pkl")

# ===============================
# Save Results
# ===============================

results_df = pd.DataFrame(results)

results_df.to_csv("classification_model_metrics.csv",index=False)

print("\nTraining complete.")
