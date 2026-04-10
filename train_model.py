import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42
NUM_ROWS = 20_000

NUMERICAL_FEATURES = [
    "amount",
    "time",
    "transaction_count_24h",
    "avg_amount_24h",
]
CATEGORICAL_FEATURES = [
    "merchant_type",
    "transaction_type",
    "device_type",
    "location_type",
]
BINARY_FEATURES = ["is_international"]
TARGET_COLUMN = "fraud"

MERCHANT_TYPES = ["food", "electronics", "travel", "grocery"]
TRANSACTION_TYPES = ["online", "POS"]
DEVICE_TYPES = ["mobile", "desktop"]
LOCATION_TYPES = ["same_city", "different_city", "international"]


def generate_synthetic_dataset(n_rows=NUM_ROWS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    merchant_type = rng.choice(MERCHANT_TYPES, size=n_rows)
    transaction_type = rng.choice(TRANSACTION_TYPES, size=n_rows)
    device_type = rng.choice(DEVICE_TYPES, size=n_rows)
    location_type = rng.choice(LOCATION_TYPES, size=n_rows)

    is_international = (location_type == "international").astype(int)

    time = np.round(rng.uniform(0, 24, size=n_rows), 2)
    transaction_count_24h = rng.poisson(3, size=n_rows)

    avg_amount_24h = np.clip(
        rng.lognormal(mean=4, sigma=0.7, size=n_rows),
        50, 15000
    )

    amount = np.clip(
        avg_amount_24h * rng.lognormal(mean=0.1, sigma=0.8, size=n_rows),
        10, 60000
    )

    fraud_probability = (
        0.02
        + (amount > 10000) * 0.1
        + (is_international == 1) * 0.2
        + ((transaction_type == "online") & (location_type == "international")) * 0.25
    )

    fraud_probability = np.clip(fraud_probability, 0.001, 0.95)
    fraud = rng.binomial(1, fraud_probability)

    return pd.DataFrame({
        "amount": amount,
        "time": time,
        "transaction_count_24h": transaction_count_24h,
        "avg_amount_24h": avg_amount_24h,
        "merchant_type": merchant_type,
        "transaction_type": transaction_type,
        "device_type": device_type,
        "location_type": location_type,
        "is_international": is_international,
        "fraud": fraud
    })


def preprocess(train_df, test_df):
    scaler = StandardScaler()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    x_train_num = scaler.fit_transform(train_df[NUMERICAL_FEATURES])
    x_test_num = scaler.transform(test_df[NUMERICAL_FEATURES])

    x_train_cat = encoder.fit_transform(train_df[CATEGORICAL_FEATURES])
    x_test_cat = encoder.transform(test_df[CATEGORICAL_FEATURES])

    x_train_bin = train_df[BINARY_FEATURES].values
    x_test_bin = test_df[BINARY_FEATURES].values

    x_train = np.hstack([x_train_num, x_train_bin, x_train_cat])
    x_test = np.hstack([x_test_num, x_test_bin, x_test_cat])

    return x_train, x_test, scaler, encoder


def train_model(output_dir="."):
    dataset = generate_synthetic_dataset()

    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    x_train, x_test, scaler, encoder = preprocess(X_train_df, X_test_df)


    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=5,
        class_weight={0:1, 1:2},
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train_res, y_train_res)

    # 🔥 Threshold tuning
    probs = model.predict_proba(x_test)[:, 1]
    threshold = 0.52
    preds = (probs >= threshold).astype(int)

    # 📊 Metrics
    print("\n--- MODEL PERFORMANCE ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, preds))

    # 📈 PR Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

    # 💾 Save model
    Path(output_dir).mkdir(exist_ok=True)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(encoder, open("encoder.pkl", "wb"))

    print("\nSaved model files successfully!")


if __name__ == "__main__":
    train_model()