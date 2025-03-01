from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


def metoda8(X_train, y_train, X_test, y_test):
    """
    Trenuje klasyfikator zespołowy Voting Classifier z użyciem RandomForest i XGBoost.

    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (lista lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (lista lub np.ndarray): Etykiety zbioru testowego.

    Zwraca:
        ensemble_model: Wytrenowany klasyfikator zespołowy.
        X_test: Cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Konwertuj macierz rzadką na gęstą, jeśli to konieczne
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Definicja klasyfikatorów bazowych
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Definicja modelu zespołowego
    ensemble_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')

    # Trening modelu zespołowego
    ensemble_model.fit(X_train, y_train)

    return ensemble_model, X_test, y_test
