# Anomalyze - Credit Card Fraud Detection using Anomaly Detection ML 

Anomalyze is a machine learning–based fraud detection system designed to identify suspicious credit card transactions using a hybrid decision approach. The system combines data-driven modeling with rule-based intelligence to improve both detection performance and explainability.

The solution integrates:

* A Random Forest classifier for probabilistic fraud detection
* A rule-based scoring engine for interpretability
* A hybrid scoring mechanism combining both signals for practical decision-making

Unlike many academic projects, this system does not rely on hidden or abstract features. Instead, it uses realistic transaction attributes and a synthetic dataset that mimics real-world fraud patterns.

---

## 🎯 Project Objective

The goal of this project is to build a clean, interpretable, and practically useful fraud detection system that:

* Uses realistic and explainable transaction features
* Avoids black-box hidden variables (e.g., V1–V28)
* Supports both single and batch transaction analysis
* Provides clear reasoning behind predictions
* Balances machine learning accuracy with business logic

---

## 📊 Model Evaluation (Final Optimized Results)

The model was improved using:

* SMOTE for class imbalance handling
* Threshold tuning for precision-recall tradeoff
* Hyperparameter tuning of Random Forest

### 🔥 Final Performance Metrics

| Metric          | Value     |
| --------------- | --------- |
| Accuracy        | **81.9%** |
| ROC-AUC         | **0.84**  |
| Fraud Precision | **39%**   |
| Fraud Recall    | **71%**   |
| Fraud F1-score  | **0.50**  |

### 📉 Confusion Matrix

|               | Predicted Normal | Predicted Fraud |
| ------------- | ---------------- | --------------- |
| Actual Normal | 2913 (TN)        | 573 (FP)        |
| Actual Fraud  | 151 (FN)         | 363 (TP)        |

---

## 🧠 Key Insight

The initial model achieved high accuracy but failed to detect fraud effectively due to class imbalance.

After optimization:

* Fraud recall improved significantly (31% → 71%)
* Precision improved with controlled false positives
* ROC-AUC increased from 0.75 → 0.84

This demonstrates the importance of prioritizing recall and precision over raw accuracy in imbalanced classification problems.

---

## ⚙️ Model Training Pipeline

1. Generate synthetic fraud dataset
2. Perform stratified train-test split
3. Scale numerical features (`StandardScaler`)
4. Encode categorical features (`OneHotEncoder`)
5. Handle class imbalance using SMOTE
6. Train Random Forest model
7. Apply threshold tuning for classification
8. Evaluate using precision, recall, F1-score, ROC-AUC

---

## 🤖 Final Model Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=3,
    class_weight={0:1, 1:3},
    random_state=42,
    n_jobs=-1
)
```

---

## 🧩 Hybrid Scoring System

The system combines ML predictions with rule-based scoring:

```text
base_score = (ml_probability * 0.55) + (rule_score / 100 * 0.45)
final_score = base_score + escalation_bonus
```

This approach improves:

* Detection of obvious fraud patterns
* Explainability for end-users
* Real-world decision reliability

---

## 📈 Why This Approach Works

Fraud detection is an imbalanced problem where:

* High accuracy can be misleading
* Missing fraud (false negatives) is costly

This project addresses that by:

* Optimizing recall for detection
* Controlling precision to reduce false alarms
* Using hybrid scoring for practical deployment

---

## 🖥️ Dashboard Features

* Single transaction analysis
* Batch CSV processing
* Risk categorization (Low / Medium / High)
* Explainable fraud reasoning
* Visual insights and summaries

---

## 📂 Project Structure

* `train_model.py` → model training pipeline
* `app.py` → Streamlit dashboard
* `model.pkl`, `scaler.pkl`, `encoder.pkl` → saved artifacts
* `requirements.txt` → dependencies

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## 📝 Final Note

This project demonstrates how combining machine learning with domain logic leads to more reliable and explainable fraud detection systems.

It reflects a practical approach aligned with real-world financial risk systems rather than purely academic accuracy-focused models.
