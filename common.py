import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, matthews_corrcoef,
    log_loss, cohen_kappa_score
)
import time

def load_and_preprocess_data():
    # Load datasets
    fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
    true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

    # Add labels
    fake_news['label'] = 0
    true_news['label'] = 1

    # Combine datasets
    data = pd.concat([fake_news, true_news]).reset_index(drop=True)

    # Features and labels
    X = data['text']
    y = data['label']

    return X, y

def load_shuffle_preprocess_data():
    # Load datasets
    fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
    true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

    # Add labels
    fake_news['label'] = 0
    true_news['label'] = 1

    # Combine datasets
    data = pd.concat([fake_news, true_news]).reset_index(drop=True)
    data = data.sample(frac=1).reset_index(drop=True)

    # Features and labels
    X = data['text']
    y = data['label']

    return X, y

def vectorize_data(X, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    return tfidf.fit_transform(X), tfidf

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def evaluate_model(model, X_test, y_test, run_type, output_path, flatten=True):
    import numpy as np
    import os
    import pandas as pd
    import time
    from sklearn.metrics import (
        f1_score,
        roc_auc_score,
        matthews_corrcoef,
        log_loss,
        cohen_kappa_score,
        precision_recall_curve,
        accuracy_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    start_time = time.time()

    # Cross-validation metrics
    cv_accuracy_mean = None
    cv_accuracy_std = None
    try:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_test, y_test, cv=skf, scoring="accuracy")
        cv_accuracy_mean = cv_scores.mean()
        cv_accuracy_std = cv_scores.std()
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_accuracy_mean:.4f}, Std Dev: {cv_accuracy_std:.4f}")
    except Exception as e:
        print(f"Could not calculate cross-validation metrics: {e}")

    # Predictions
    if hasattr(model, "predict_proba"):  # For scikit-learn models
        y_pred_proba = model.predict_proba(X_test)
        # Handle single-column predict_proba case
        if y_pred_proba.shape[1] == 1:
            y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
        y_pred_proba = y_pred_proba[:, 1]  # Use probabilities for the positive class
    else:  # For Keras Sequential models
        if flatten:
            y_pred_proba = model.predict(X_test).flatten()
        else:
            y_pred_proba = model.predict(X_test)

    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Initialize metrics dictionary
    metrics = {}

    # Metrics calculation with checks
    try:
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    except ValueError:
        metrics["Accuracy"] = None

    try:
        metrics["Precision"] = precision_score(y_test, y_pred, zero_division=0)
    except ValueError:
        metrics["Precision"] = None

    try:
        metrics["Recall"] = recall_score(y_test, y_pred, zero_division=0)
    except ValueError:
        metrics["Recall"] = None

    try:
        metrics["F1-Score"] = f1_score(y_test, y_pred)
    except ValueError:
        metrics["F1-Score"] = None

    try:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        metrics["ROC-AUC"] = None

    try:
        metrics["MCC"] = matthews_corrcoef(y_test, y_pred)
    except ValueError:
        metrics["MCC"] = None

    try:
        metrics["Log Loss"] = log_loss(y_test, y_pred_proba)
    except ValueError:
        metrics["Log Loss"] = None

    try:
        metrics["Cohen's Kappa"] = cohen_kappa_score(y_test, y_pred)
    except ValueError:
        metrics["Cohen's Kappa"] = None

    # Execution time
    metrics["Execution Time (s)"] = time.time() - start_time

    # Add cross-validation metrics
    metrics["CV Accuracy (Mean)"] = cv_accuracy_mean
    metrics["CV Accuracy (Std Dev)"] = cv_accuracy_std

    # Save metrics
    results_file = os.path.join(output_path, f"{run_type}_results.csv")
    pd.DataFrame([metrics]).to_csv(results_file, index=False)
    print(f"Saved results for {run_type}")

    # Precision-Recall curve
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_curve_file = os.path.join(output_path, f"{run_type}_pr_curve.csv")
        pd.DataFrame({"Precision": precision, "Recall": recall}).to_csv(pr_curve_file, index=False)
        print(f"Saved Precision-Recall curve for {run_type}")
    except ValueError:
        print(f"Could not calculate Precision-Recall curve for {run_type}")

    # Print metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")

