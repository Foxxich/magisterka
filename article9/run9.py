from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from common import load_and_preprocess_data, vectorize_data, split_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_run9():
    # Krok 1: Wczytaj i przetwórz zbiór danych
    X, y = load_and_preprocess_data()

    # Krok 2: Wektoryzacja danych tekstowych za pomocą TF-IDF
    X_tfidf, _ = vectorize_data(X, max_features=5000)

    # Krok 3: Podział danych na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.3)

    # Krok 4: Definicja klasyfikatorów bazowych z dostosowanymi hiperparametrami
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    svc = SVC(kernel='linear', probability=True, random_state=42)

    # Krok 5: Definicja metod zespołowych
    bagging = BaggingClassifier(estimator=rf, n_estimators=50, random_state=42)
    boosting = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    voting = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('svc', svc)], voting='soft')

    # Krok 6: Trenowanie modeli i ocena
    models = {
        "Random Forest": rf,
        "Bagging": bagging,
        "Boosting (AdaBoost)": boosting,
        "Voting Classifier": voting
    }

    results = {}
    for model_name, model in models.items():
        print(f"Trenowanie {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
        report = classification_report(y_test, y_pred, output_dict=True)
        results[model_name] = {
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
            "Classification Report": report
        }
        print(f"Dokładność dla {model_name}: {accuracy}")
        if roc_auc:
            print(f"ROC-AUC dla {model_name}: {roc_auc}")
        print(classification_report(y_test, y_pred))

    # Zwracanie najlepszego modelu
    return voting, X_test, y_test