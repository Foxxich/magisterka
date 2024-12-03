import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_run17(X_train, y_train, X_test, y_test, n_models=5):
    """
    Trenuje sieć neuronową Sequential przy użyciu ważenia próbek i zwraca model oraz dane testowe.
    """
    sample_weights = np.ones(len(y_train))  # Inicjalizacja wag próbek
    for _ in range(n_models):
        model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation=None, kernel_regularizer=l2(0.01)),  # Warstwa ukryta 1
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(64, activation=None, kernel_regularizer=l2(0.01)),  # Warstwa ukryta 2
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Warstwa wyjściowa
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=64, sample_weight=sample_weights, verbose=0)
        y_pred = model.predict(X_train)
        sample_weights += np.abs(y_train - y_pred.squeeze())  # Aktualizacja wag próbek
    return model, X_test, y_test

def train_run18(X_train, y_train, X_test, y_test):
    """
    Trenuje klasyfikatory GradientBoosting i CatBoost oraz łączy ich predykcje przy użyciu meta-modelu.
    """
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Gradient Boosting
    cat_clf = CatBoostClassifier(iterations=100, verbose=0, random_state=42)  # CatBoost
    gb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)
    gb_preds_train = gb_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla GradientBoosting
    cat_preds_train = cat_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla CatBoost
    meta_features_train = pd.DataFrame({  # Tworzenie meta-cech
        'gb_preds': gb_preds_train,
        'cat_preds': cat_preds_train
    })
    meta_model = LogisticRegression()  # Meta-model: regresja logistyczna
    meta_model.fit(meta_features_train, y_train)
    gb_preds_test = gb_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla GradientBoosting
    cat_preds_test = cat_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla CatBoost
    meta_features_test = pd.DataFrame({  # Meta-cechy dla testu
        'gb_preds': gb_preds_test,
        'cat_preds': cat_preds_test
    })
    return meta_model, meta_features_test, y_test

def train_run19(X_train, y_train, X_test, y_test):
    """
    Trenuje klasyfikatory RandomForest i ExtraTrees oraz łączy ich predykcje przy użyciu meta-modelu.
    """
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest
    et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)  # Extra Trees
    rf_clf.fit(X_train, y_train)
    et_clf.fit(X_train, y_train)
    rf_preds_train = rf_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla Random Forest
    et_preds_train = et_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla Extra Trees
    meta_features_train = pd.DataFrame({  # Tworzenie meta-cech
        'rf_preds': rf_preds_train,
        'et_preds': et_preds_train
    })
    meta_model = LogisticRegression()  # Meta-model: regresja logistyczna
    meta_model.fit(meta_features_train, y_train)
    rf_preds_test = rf_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla Random Forest
    et_preds_test = et_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla Extra Trees
    meta_features_test = pd.DataFrame({  # Meta-cechy dla testu
        'rf_preds': rf_preds_test,
        'et_preds': et_preds_test
    })
    return meta_model, meta_features_test, y_test

def train_run20(X_train, y_train, X_test, y_test):
    """
    Trenuje klasyfikatory LightGBM i CatBoost oraz łączy ich predykcje przy użyciu meta-modelu.
    """
    lgb_clf = LGBMClassifier(n_estimators=100, random_state=42)  # LightGBM
    cat_clf = CatBoostClassifier(iterations=100, verbose=0, random_state=42)  # CatBoost
    lgb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)
    lgb_preds_train = lgb_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla LightGBM
    cat_preds_train = cat_clf.predict_proba(X_train)[:, 1]  # Predykcje treningowe dla CatBoost
    meta_features_train = pd.DataFrame({  # Tworzenie meta-cech
        'lgb_preds': lgb_preds_train,
        'cat_preds': cat_preds_train
    })
    meta_model = LogisticRegression()  # Meta-model: regresja logistyczna
    meta_model.fit(meta_features_train, y_train)
    lgb_preds_test = lgb_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla LightGBM
    cat_preds_test = cat_clf.predict_proba(X_test)[:, 1]  # Predykcje testowe dla CatBoost
    meta_features_test = pd.DataFrame({  # Meta-cechy dla testu
        'lgb_preds': lgb_preds_test,
        'cat_preds': cat_preds_test
    })
    return meta_model, meta_features_test, y_test
