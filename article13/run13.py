from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from common import load_and_preprocess_data, evaluate_model


def train_run13():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Podziel zbiór danych na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Konwertuj dane tekstowe na cechy TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Radzenie sobie z niezrównoważonymi klasami za pomocą SMOTETomek
    smotetomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train_tfidf, y_train)

    # Zdefiniuj i wytrenuj model XGBoost
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train_resampled) / np.sum(y_train_resampled == 1),
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        random_state=42
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Zwróć model i dane testowe do oceny
    return xgb_model, X_test_tfidf, y_test
