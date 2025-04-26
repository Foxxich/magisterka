import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from transformers import (BertTokenizer, RobertaTokenizer,
                          TFBertForSequenceClassification,
                          TFRobertaForSequenceClassification)


def load_and_preprocess_data(dataset_input):
    # Wczytanie zbiorów danych
    project_root = os.getcwd()

    # Build paths to the dataset files
    if dataset_input == "ISOT":
        fake_news_path = os.path.join(project_root, "datasets", "ISOT_dataset", "Fake.csv")
        true_news_path = os.path.join(project_root, "datasets", "ISOT_dataset", "True.csv")
    elif dataset_input == "BuzzFeed":
        fake_news_path = os.path.join(project_root, "datasets", "BuzzFeed_dataset", "Fake.csv")
        true_news_path = os.path.join(project_root, "datasets", "BuzzFeed_dataset", "True.csv")
    else:
        fake_news_path = os.path.join(project_root, "datasets", "WELFake_dataset", "Fake.csv")
        true_news_path = os.path.join(project_root, "datasets", "WELFake_dataset", "True.csv")

    # Load the datasets
    fake_news = pd.read_csv(fake_news_path)
    true_news = pd.read_csv(true_news_path)

    # Dodanie etykiet
    fake_news['label'] = 0
    true_news['label'] = 1

    # Połączenie zbiorów danych
    data = pd.concat([fake_news, true_news]).reset_index(drop=True)

    # Funkcje i etykiety
    X = data['text']
    y = data['label']

    return X, y

def get_bert_embeddings(texts, batch_size=32, max_length=128, num_labels=2):
    # Preprocess texts: convert to strings, handle NaN/None/float values
    cleaned_texts = []
    for text in texts:
        if isinstance(text, (str, bytes)):
            cleaned_texts.append(text)
        elif pd.isna(text) or text is None or isinstance(text, (float, int)):
            cleaned_texts.append("")  # Replace NaN/None/float with empty string
        else:
            cleaned_texts.append(str(text))  # Convert other types to string

    # Tokenizacja tekstów dla modelu BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
    )

    # Tokenizacja wszystkich tekstów w jednej operacji dla oszczędności czasu
    inputs = tokenizer(
        cleaned_texts, return_tensors="tf", padding=True, truncation=True, max_length=max_length
    )

    # Przetwarzanie w partiach
    embeddings = []
    for i in range(0, len(cleaned_texts), batch_size):
        batch_inputs = {
            k: v[i:i + batch_size] for k, v in inputs.items()
        }
        outputs = bert_model(**batch_inputs)
        embeddings.append(outputs.logits.numpy())

    return np.vstack(embeddings)

def get_roberta_embeddings(texts, batch_size=32, max_length=128, num_labels=2):
    # Preprocess texts: convert to strings, handle NaN/None/float values
    cleaned_texts = []
    for text in texts:
        if isinstance(text, (str, bytes)):
            cleaned_texts.append(text)
        elif pd.isna(text) or text is None or isinstance(text, (float, int)):
            cleaned_texts.append("")  # Replace NaN/None/float with empty string
        else:
            cleaned_texts.append(str(text))  # Convert other types to string

    # Tokenizacja tekstów dla modelu RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = TFRobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels,
    )

    # Tokenizacja wszystkich tekstów w jednej operacji dla oszczędności czasu
    inputs = tokenizer(
        cleaned_texts, return_tensors="tf", padding=True, truncation=True, max_length=max_length
    )

    # Przetwarzanie w partiach
    embeddings = []
    for i in range(0, len(cleaned_texts), batch_size):
        batch_inputs = {
            k: v[i:i + batch_size] for k, v in inputs.items()
        }
        outputs = roberta_model(**batch_inputs)
        embeddings.append(outputs.logits.numpy())

    return np.vstack(embeddings)

def get_transformer_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    
    # Preprocess texts: convert to strings, handle NaN/None/float values
    cleaned_texts = []
    for text in texts:
        # Skip or convert invalid values to empty string
        if isinstance(text, (str, bytes)):
            cleaned_texts.append(text)
        elif pd.isna(text) or text is None or isinstance(text, (float, int)):
            cleaned_texts.append("")  # Replace NaN/None/float with empty string
        else:
            cleaned_texts.append(str(text))  # Convert other types to string
    
    # Generate embeddings
    embeddings = model.encode(cleaned_texts, show_progress_bar=True)
    
    return np.array(embeddings)

def vectorize_data(X, max_features=5000):
    # Wejściowe dane tekstowe są przekształcane za pomocą TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    return tfidf.fit_transform(X), tfidf

def split_data(X, y, test_size=0.2):
    # Podział danych na zbiory treningowe i testowe
    return train_test_split(X, y, test_size=test_size, random_state=42)

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

def evaluate_model(model, X_test, y_test, run_type, output_path, start_time, dataset_input, flatten=True):
    # Metryki walidacji krzyżowej
    cv_accuracy_mean = None
    cv_accuracy_std = None
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_test, y_test, cv=skf, scoring="accuracy")
        cv_accuracy_mean = cv_scores.mean()
        cv_accuracy_std = cv_scores.std()
        print(f"Wyniki dokładności walidacji krzyżowej: {cv_scores}")
        print(f"Średnia dokładność CV: {cv_accuracy_mean:.4f}, Odchylenie standardowe: {cv_accuracy_std:.4f}")
    except Exception as e:
        print(f"Nie można obliczyć metryk walidacji krzyżowej: {e}")

    # Predykcje
    if hasattr(model, "predict_proba"):  # Dla modeli scikit-learn
        y_pred_proba = model.predict_proba(X_test)
        # Obsługa przypadku pojedynczej kolumny predict_proba
        if y_pred_proba.shape[1] == 1:
            y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
        y_pred_proba = y_pred_proba[:, 1]  # Użyj prawdopodobieństw dla klasy pozytywnej
    else:  # Dla modeli Sequential w Keras
        if flatten:
            y_pred_proba = model.predict(X_test).flatten()
        else:
            y_pred_proba = model.predict(X_test)

    # Konwersja prawdopodobieństw na predykcje binarne
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Inicjalizacja słownika metryk
    metric_functions = {
        "Accuracy": (accuracy_score, {"y_true": y_test, "y_pred": y_pred}),
        "Precision": (precision_score, {"y_true": y_test, "y_pred": y_pred, "zero_division": 0}),
        "Recall": (recall_score, {"y_true": y_test, "y_pred": y_pred, "zero_division": 0}),
        "F1-Score": (f1_score, {"y_true": y_test, "y_pred": y_pred}),
        "ROC-AUC": (roc_auc_score, {"y_true": y_test, "y_score": y_pred_proba}),
        "MCC": (matthews_corrcoef, {"y_true": y_test, "y_pred": y_pred}),
        "Log Loss": (log_loss, {"y_true": y_test, "y_pred": y_pred_proba}),
        "Cohen's Kappa": (cohen_kappa_score, {"y1": y_test, "y2": y_pred})
    }

    # Obliczanie metryk z obsługą wyjątków
    metrics = {}
    for metric_name, (func, params) in metric_functions.items():
        try:
            metrics[metric_name] = func(**params)
        except ValueError:
            metrics[metric_name] = None

    # Czas wykonania
    metrics["Execution Time (s)"] = time.time() - start_time

    # Dodanie metryk walidacji krzyżowej
    metrics["CV Accuracy (Mean)"] = cv_accuracy_mean
    metrics["CV Accuracy (Std Dev)"] = cv_accuracy_std

    # Zapis metryk
    results_file = os.path.join(output_path, f"{run_type}_{dataset_input}_results.csv")
    pd.DataFrame([metrics]).to_csv(results_file, index=False)
    print(f"Zapisano wyniki dla {run_type}")

    # Krzywa Precision-Recall
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_curve_file = os.path.join(output_path, f"{run_type}_{dataset_input}_pr_curve.csv")
        pd.DataFrame({"Precision": precision, "Recall": recall}).to_csv(pr_curve_file, index=False)
        print(f"Zapisano krzywą Precision-Recall dla {run_type}")
    except ValueError:
        print(f"Nie można obliczyć krzywej Precision-Recall dla {run_type}")

    # Wyświetlenie metryk
    print("Metryki oceny:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")