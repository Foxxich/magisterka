import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import numpy as np
import os
from transformers import BertTokenizer
from transformers import RobertaTokenizer
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

from transformers import BertTokenizer, TFBertForSequenceClassification

from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

def get_bert_embeddings(texts, batch_size=32, max_length=128, num_labels=2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
    )

    # Tokenize all texts in a single call to save time
    inputs = tokenizer(
        texts, return_tensors="tf", padding=True, truncation=True, max_length=max_length
    )

    # Process in batches
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_inputs = {
            k: v[i:i + batch_size] for k, v in inputs.items()
        }
        outputs = bert_model(**batch_inputs)
        embeddings.append(outputs.logits.numpy())

    return np.vstack(embeddings)

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

def get_roberta_embeddings(texts, batch_size=32, max_length=128, num_labels=2):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = TFRobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels,
    )

    # Tokenize all texts in a single call to save time
    inputs = tokenizer(
        texts, return_tensors="tf", padding=True, truncation=True, max_length=max_length
    )

    # Process in batches
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_inputs = {
            k: v[i:i + batch_size] for k, v in inputs.items()
        }
        outputs = roberta_model(**batch_inputs)
        embeddings.append(outputs.logits.numpy())

    return np.vstack(embeddings)

from sentence_transformers import SentenceTransformer
import numpy as np

def get_transformer_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings using Sentence Transformers.
    Args:
        texts (list of str): Input texts to embed.
        model_name (str): Pretrained Sentence Transformer model name.
    Returns:
        np.ndarray: Embeddings for the given texts.
    """
    # Load the Sentence Transformer model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    return np.array(embeddings)

def vectorize_data(X, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    return tfidf.fit_transform(X), tfidf

def split_data(X, y, unseen_class_count=1):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Wyznacz widoczne i niewidoczne klasy
    classes = y.unique()
    unseen_classes = classes[-unseen_class_count:]  # Ostatnie klasy jako niewidoczne
    seen_mask = ~y.isin(unseen_classes)
    unseen_mask = ~seen_mask

    # Podział na zbiór treningowy (widoczne klasy) i testowy (niewidoczne klasy)
    X_train = X[seen_mask].reset_index(drop=True)
    y_train = y[seen_mask].reset_index(drop=True)
    X_test = X[unseen_mask].reset_index(drop=True)
    y_test = y[unseen_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

def split_data_one_shot(X, y):
    """Podział danych na podstawie jednego przykładu na klasę."""
    # Konwersja do Pandas DataFrame/Series w razie potrzeby
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Wybór jednego przykładu na klasę
    X_test = X.groupby(y).apply(lambda group: group.iloc[0]).reset_index(drop=True)
    y_test = y.groupby(y).first().reset_index(drop=True)

    # Reszta danych jako treningowe
    test_indices = y.groupby(y).apply(lambda group: group.index[0]).values
    train_mask = ~X.index.isin(test_indices)

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def split_data_few_shot(X, y, few_shot_examples=5):
    """Podział danych na podstawie kilku przykładów na klasę."""
    # Konwersja do Pandas DataFrame/Series w razie potrzeby
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Wybór kilku przykładów na klasę
    sampled_data = y.groupby(y).apply(lambda group: group.sample(n=min(few_shot_examples, len(group))))
    test_indices = sampled_data.index.get_level_values(1)

    X_test = X.loc[test_indices].reset_index(drop=True)
    y_test = y.loc[test_indices].reset_index(drop=True)

    # Reszta danych jako treningowe
    train_mask = ~X.index.isin(test_indices)
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, run_type, output_path, start_time, flatten=True):
    # Cross-validation metrics
    cv_accuracy_mean = None
    cv_accuracy_std = None
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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