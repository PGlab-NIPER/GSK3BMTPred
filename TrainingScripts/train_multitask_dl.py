#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multitask Deep Learning Model for GSK3β Inhibitor Prediction

This script trains a multitask neural network for:
1) Classification (Inhibitor vs Non-inhibitor)
2) Regression (pIC50 prediction)

Training uses stratified 10-fold cross-validation.

Author: Pradnya Kamble
"""

# ==============================
# Imports
# ==============================

import os
import random
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)

from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping


# ==============================
# Reproducibility
# ==============================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ==============================
# Parameters
# ==============================

BATCH_SIZE = 64
EPOCHS = 2000
KFOLDS = 10
LEARNING_RATE = 0.00002


# ==============================
# Model Architecture
# ==============================

def build_multitask_model(input_dim):

    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(
        1000,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(0.001),
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        50,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(0.001),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        5,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(0.001),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    class_output = layers.Dense(1, activation="sigmoid", name="classification")(x)
    reg_output = layers.Dense(1, activation="linear", name="regression")(x)

    model = models.Model(inputs=inputs, outputs=[class_output, reg_output])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "classification": "binary_crossentropy",
            "regression": "mse",
        },
        metrics={
            "classification": "accuracy",
            "regression": "mae",
        },
    )

    return model


# ==============================
# Evaluation Function
# ==============================

def evaluate_classification(y_true, y_prob):

    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def evaluate_regression(y_true, y_pred):

    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ==============================
# Main Training Function
# ==============================

def main():

    data = pd.read_csv("gsk3b_padel_desc_cleaned.csv")

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data.drop(columns=["Compound_CID", "SMILES", "Class", "IC50_microM", "pic50"])

    y_class = data["Class"]
    y_reg = data["pic50"]

    # Train/test split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.2,
        stratify=y_class,
        random_state=SEED,
    )

    # Scaling
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "padel_scaler.pkl")

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_class_train),
        y=y_class_train,
    )

    class_weight_dict = dict(enumerate(class_weights))

    # Cross-validation
    skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_class_train)):

        print(f"\n===== Fold {fold+1} =====")

        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]

        y_class_tr = y_class_train.iloc[train_idx]
        y_class_val = y_class_train.iloc[val_idx]

        y_reg_tr = y_reg_train.iloc[train_idx]
        y_reg_val = y_reg_train.iloc[val_idx]

        model = build_multitask_model(X.shape[1])

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
        )

        model.fit(
            X_tr,
            [y_class_tr, y_reg_tr],
            validation_data=(X_val, [y_class_val, y_reg_val]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )

        class_prob, reg_pred = model.predict(X_val)

        class_metrics = evaluate_classification(y_class_val, class_prob.ravel())
        reg_metrics = evaluate_regression(y_reg_val, reg_pred.ravel())

        fold_results.append({**class_metrics, **reg_metrics})

    results_df = pd.DataFrame(fold_results)

    print("\n===== Cross Validation Results =====")
    print(results_df.mean())
    print(results_df.std())

    # Train final model on full training set
    final_model = build_multitask_model(X.shape[1])

    final_model.fit(
        X_train,
        [y_class_train, y_reg_train],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    final_model.save("gsk3b_multitask_model.keras")
    final_model.save("gsk3b_multitask_model.h5")

    # Test evaluation
    class_prob, reg_pred = final_model.predict(X_test)

    class_metrics = evaluate_classification(y_class_test, class_prob.ravel())
    reg_metrics = evaluate_regression(y_reg_test, reg_pred.ravel())

    print("\n===== Test Performance =====")
    print({**class_metrics, **reg_metrics})


# ==============================
# Run
# ==============================

if __name__ == "__main__":
    main()
