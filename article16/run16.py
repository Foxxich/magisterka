import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common import load_and_preprocess_data, vectorize_data, split_data, evaluate_model

def preprocess_text(X):
    """
    Niestandardowe przetwarzanie: zamiana na małe litery, usuwanie stopwords i stemming.
    """
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    import re

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)  # Usuń adresy URL
        text = re.sub(r"[^\w\s]", "", text)  # Usuń znaki interpunkcyjne
        text = re.sub(r"\d+", "", text)  # Usuń liczby
        words = text.lower().split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    return X.apply(clean_text)

def train_run16():
    # Krok 1: Wczytaj i przetwórz zbiór danych
    X, y = load_and_preprocess_data()
    X = preprocess_text(X)

    # Krok 2: Wektoryzacja danych tekstowych za pomocą TF-IDF
    X_tfidf, vectorizer = vectorize_data(X, max_features=5000)

    # Krok 3: Podział danych na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.2)

    # Krok 4: Definicja modeli bazowych
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svc_model = SVC(probability=True, kernel="linear", random_state=42)

    # Krok 5: Generowanie meta-cech dla klasyfikatora stacking
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_preds_train = cross_val_predict(rf_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    gb_preds_train = cross_val_predict(gb_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    svc_preds_train = cross_val_predict(svc_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]

    meta_features_train = np.column_stack((rf_preds_train, gb_preds_train, svc_preds_train))

    # Krok 6: Trenowanie meta-klasyfikatora z użyciem regresji logistycznej
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(meta_features_train, y_train)

    # Krok 7: Dopasowanie modeli bazowych na pełnym zbiorze treningowym
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)

    # Krok 8: Generowanie meta-cech dla zbioru testowego
    rf_preds_test = rf_model.predict_proba(X_test)[:, 1]
    gb_preds_test = gb_model.predict_proba(X_test)[:, 1]
    svc_preds_test = svc_model.predict_proba(X_test)[:, 1]
    meta_features_test = np.column_stack((rf_preds_test, gb_preds_test, svc_preds_test))

    # Krok 9: Zwrócenie wytrenowanego meta-modelu oraz danych testowych
    return meta_model, meta_features_test, y_test
