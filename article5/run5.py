from common import load_and_preprocess_data, vectorize_data
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_run5(X_train, y_train, X_test, y_test):
    """Trenuje model stacking z użyciem Random Forest i XGBoost jako bazowych oraz Logistic Regression jako meta-modelu."""
    
    # Trenuj modele bazowe: RandomForest i XGBoost
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)

    rf_clf.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)

    # Uzyskaj predykcje dla zbioru treningowego, aby stworzyć cechy dla meta-modelu
    rf_preds_train = rf_clf.predict_proba(X_train)[:, 1]
    xgb_preds_train = xgb_clf.predict_proba(X_train)[:, 1]

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

    # Zwróć model meta i cechy testowe do oceny
    return meta_model, meta_features_test, y_test
