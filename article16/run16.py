import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold


def preprocess_text(X):
    """
    Niestandardowe przetwarzanie tekstu: konwersja do małych liter, usuwanie stop-słów i zastosowanie stemmingu.
    """
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    import re

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    def clean_text(text):
        text = re.sub(r"http\S+", "", text)  # Usuwanie URL-i
        text = re.sub(r"[^\w\s]", "", text)  # Usuwanie znaków interpunkcyjnych
        text = re.sub(r"\d+", "", text)  # Usuwanie liczb
        words = text.lower().split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    return X.apply(clean_text)


def metoda16(X_train, y_train, X_test, y_test):
    """
    Trenuje model zespołowy stacking z meta-klasyfikatorem używającym regresji logistycznej.

    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (np.ndarray): Etykiety zbioru testowego.

    Zwraca:
        meta_model: Wytrenowany meta-klasyfikator (stacking).
        meta_features_test: Meta-cechy dla zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Inicjalizacja modeli bazowych
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svc_model = SVC(probability=True, kernel="linear", random_state=42)

    # Generowanie meta-cech dla treningu przy użyciu walidacji krzyżowej
    print("Generowanie meta-cech dla treningu...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_preds_train = cross_val_predict(rf_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    gb_preds_train = cross_val_predict(gb_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    svc_preds_train = cross_val_predict(svc_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]

    meta_features_train = np.column_stack((rf_preds_train, gb_preds_train, svc_preds_train))

    # Trening meta-klasyfikatora używając regresji logistycznej
    print("Trening meta-klasyfikatora...")
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(meta_features_train, y_train)

    # Trening modeli bazowych na pełnych danych treningowych
    print("Trening modeli bazowych...")
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)

    # Generowanie meta-cech dla testu
    print("Generowanie meta-cech dla testu...")
    rf_preds_test = rf_model.predict_proba(X_test)[:, 1]
    gb_preds_test = gb_model.predict_proba(X_test)[:, 1]
    svc_preds_test = svc_model.predict_proba(X_test)[:, 1]
    meta_features_test = np.column_stack((rf_preds_test, gb_preds_test, svc_preds_test))

    # Ocena i wyświetlenie wyników modelu
    print("Ocena wyników modelu...")
    y_pred = meta_model.predict(meta_features_test)
    print(classification_report(y_test, y_pred))
    print(f"Dokładność: {accuracy_score(y_test, y_pred):.4f}")

    return meta_model, meta_features_test, y_test
