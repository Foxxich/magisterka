from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


def train_run12(X_train, y_train, X_test, y_test):
    """
    Trenuje i ocenia klasyfikatory Random Forest oraz CatBoost.

    Parametry:
        X_train (array-like): Cechy zbioru treningowego.
        y_train (array-like): Etykiety zbioru treningowego.
        X_test (array-like): Cechy zbioru testowego.
        y_test (array-like): Etykiety zbioru testowego.

    Zwraca:
        dict: Słownik zawierający wytrenowane modele oraz odpowiadające im dane testowe.
    """
    # Upewnij się, że dane wejściowe są 2-wymiarowe
    if len(X_train.shape) != 2 or len(X_test.shape) != 2:
        raise ValueError("Cechy wejściowe muszą być 2-wymiarowymi tablicami.")

    # Trening Random Forest (Bagging)
    print("Trening Random Forest...")
    rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Trening CatBoost (Boosting)
    print("Trening CatBoost...")
    catboost_classifier = CatBoostClassifier(
        iterations=200,
        learning_rate=0.01,
        eval_metric='Accuracy',
        early_stopping_rounds=20,
        use_best_model=True,
        verbose=50
    )
    catboost_classifier.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

    # Zwróć oba modele wraz z danymi testowymi do oceny
    return {
        "RandomForest": (rf_classifier, X_test, y_test),
        "CatBoost": (catboost_classifier, X_test, y_test)
    }
