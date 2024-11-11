from common import load_and_preprocess_data, vectorize_data, split_data
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_run5():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Wektoryzacja danych
    X_vec, _ = vectorize_data(X, max_features=5000)

    # Podziel dane na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_vec, y, test_size=0.2)

    # Trenuj modele bazowe: RandomForest i XGBoost
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)

    rf_clf.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)

    # Uzyskaj predykcje walidacyjne do modelu meta (stacking)
    rf_preds_train = cross_val_predict(rf_clf, X_train, y_train, method="predict_proba")[:, 1]
    xgb_preds_train = cross_val_predict(xgb_clf, X_train, y_train, method="predict_proba")[:, 1]

    # Połącz predykcje dla modelu meta
    meta_features_train = pd.DataFrame({
        'rf_preds': rf_preds_train,
        'xgb_preds': xgb_preds_train
    })

    # Trenuj model meta
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, y_train)

    # Wygeneruj predykcje dla zbioru testowego
    rf_preds_test = rf_clf.predict_proba(X_test)[:, 1]
    xgb_preds_test = xgb_clf.predict_proba(X_test)[:, 1]
    meta_features_test = pd.DataFrame({
        'rf_preds': rf_preds_test,
        'xgb_preds': xgb_preds_test
    })

    # Zwróć model meta i dane testowe (meta_features_test zamiast X_test)
    return meta_model, meta_features_test, y_test
