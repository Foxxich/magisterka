from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # Zastąpiono SVC na KNN
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
)


def metoda9(X_train, y_train, X_test, y_test):
    """
    Trenuje różne modele zespołowe i ocenia ich wyniki.
    
    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (lista lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (lista lub np.ndarray): Etykiety zbioru testowego.
    
    Zwraca:
        voting: Najlepszy model (VotingClassifier).
        X_test: Cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Definicja klasyfikatorów bazowych
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)  # Zastąpiono SVC na KNN z domyślnym n_neighbors=5

    # Definicja metod zespołowych
    bagging = BaggingClassifier(estimator=rf, n_estimators=50, random_state=42)
    boosting = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    voting = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('knn', knn)], voting='soft')

    # Trenuj modele i oceniaj
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
        results[model_name] = {
            "Dokładność": accuracy,
            "ROC-AUC": roc_auc,
            "Raport klasyfikacji": classification_report(y_test, y_pred, output_dict=True)
        }
        print(f"Dokładność dla {model_name}: {accuracy}")
        if roc_auc is not None:
            print(f"ROC-AUC dla {model_name}: {roc_auc}")
        print(classification_report(y_test, y_pred))

    # Zwróć najlepszy model (Voting Classifier) i dane testowe
    return model, X_test, y_test