from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np

def train_run17(X_train, y_train, X_test, y_test, n_models=5):
    print("\n=== Boosting (train_run17) ===")
    sample_weights = np.ones(len(y_train))
    for _ in range(n_models):
        model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation=None, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(64, activation=None, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=64, sample_weight=sample_weights, verbose=0)
        y_pred = model.predict(X_train)
        sample_weights += np.abs(y_train - y_pred.squeeze())  # Adjust sample weights based on errors
    
    return model, X_test, y_test

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re


import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common import vectorize_data, split_data


def preprocess_text(X):
    """
    Custom text preprocessing: convert to lowercase, remove stopwords, and apply stemming.
    """
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    import re

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    def clean_text(text):
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\d+", "", text)  # Remove numbers
        words = text.lower().split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    return X.apply(clean_text)

from common import load_and_preprocess_data, vectorize_data
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_run18(X_train, y_train, X_test, y_test):
    """Trenuje model stacking z użyciem Gradient Boosting i CatBoost jako bazowych oraz Logistic Regression jako meta-modelu."""
    
    # Trenuj modele bazowe: GradientBoosting i CatBoost
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    cat_clf = CatBoostClassifier(iterations=100, verbose=0, random_state=42)

    gb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)

    # Uzyskaj predykcje dla zbioru treningowego, aby stworzyć cechy dla meta-modelu
    gb_preds_train = gb_clf.predict_proba(X_train)[:, 1]
    cat_preds_train = cat_clf.predict_proba(X_train)[:, 1]

    # Połącz predykcje dla modelu meta
    meta_features_train = pd.DataFrame({
        'gb_preds': gb_preds_train,
        'cat_preds': cat_preds_train
    })

    # Trenuj model meta
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, y_train)

    # Wygeneruj predykcje dla zbioru testowego
    gb_preds_test = gb_clf.predict_proba(X_test)[:, 1]
    cat_preds_test = cat_clf.predict_proba(X_test)[:, 1]
    meta_features_test = pd.DataFrame({
        'gb_preds': gb_preds_test,
        'cat_preds': cat_preds_test
    })

    # Zwróć model meta i cechy testowe do oceny
    return meta_model, meta_features_test, y_test


from common import load_and_preprocess_data, vectorize_data
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_run19(X_train, y_train, X_test, y_test):
    """Trenuje model stacking z użyciem Random Forest i Extra Trees jako bazowych oraz Logistic Regression jako meta-modelu."""
    
    # Trenuj modele bazowe: RandomForest i ExtraTrees
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)

    rf_clf.fit(X_train, y_train)
    et_clf.fit(X_train, y_train)

    # Uzyskaj predykcje dla zbioru treningowego, aby stworzyć cechy dla meta-modelu
    rf_preds_train = rf_clf.predict_proba(X_train)[:, 1]
    et_preds_train = et_clf.predict_proba(X_train)[:, 1]

    # Połącz predykcje dla modelu meta
    meta_features_train = pd.DataFrame({
        'rf_preds': rf_preds_train,
        'et_preds': et_preds_train
    })

    # Trenuj model meta
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, y_train)

    # Wygeneruj predykcje dla zbioru testowego
    rf_preds_test = rf_clf.predict_proba(X_test)[:, 1]
    et_preds_test = et_clf.predict_proba(X_test)[:, 1]
    meta_features_test = pd.DataFrame({
        'rf_preds': rf_preds_test,
        'et_preds': et_preds_test
    })

    # Zwróć model meta i cechy testowe do oceny
    return meta_model, meta_features_test, y_test


from common import load_and_preprocess_data, vectorize_data
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_run20(X_train, y_train, X_test, y_test):
    """Trenuje model stacking z użyciem LightGBM i CatBoost jako bazowych oraz Logistic Regression jako meta-modelu."""
    
    # Trenuj modele bazowe: LightGBM i CatBoost
    lgb_clf = LGBMClassifier(n_estimators=100, random_state=42)
    cat_clf = CatBoostClassifier(iterations=100, verbose=0, random_state=42)

    lgb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)

    # Uzyskaj predykcje dla zbioru treningowego, aby stworzyć cechy dla meta-modelu
    lgb_preds_train = lgb_clf.predict_proba(X_train)[:, 1]
    cat_preds_train = cat_clf.predict_proba(X_train)[:, 1]

    # Połącz predykcje dla modelu meta
    meta_features_train = pd.DataFrame({
        'lgb_preds': lgb_preds_train,
        'cat_preds': cat_preds_train
    })

    # Trenuj model meta
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, y_train)

    # Wygeneruj predykcje dla zbioru testowego
    lgb_preds_test = lgb_clf.predict_proba(X_test)[:, 1]
    cat_preds_test = cat_clf.predict_proba(X_test)[:, 1]
    meta_features_test = pd.DataFrame({
        'lgb_preds': lgb_preds_test,
        'cat_preds': cat_preds_test
    })

    # Zwróć model meta i cechy testowe do oceny
    return meta_model, meta_features_test, y_test

