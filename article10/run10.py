from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_run10(X_train, y_train, X_test, y_test):
    """
    Trenuje model zespołowy z miękkim głosowaniem (soft-voting) i ocenia jego wyniki.

    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (lista lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (lista lub np.ndarray): Etykiety zbioru testowego.

    Zwraca:
        voting_clf: Wytrenowany model VotingClassifier.
        X_test: Cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Inicjalizacja klasyfikatorów z dostosowanymi hiperparametrami
    mlp = MLPClassifier(alpha=0.01, hidden_layer_sizes=(14,), max_iter=100, solver='lbfgs', random_state=0)
    log_reg = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=100, random_state=0)
    xgb = XGBClassifier(
        gamma=1, learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=100,
        use_label_encoder=False, eval_metric='logloss'
    )

    # Utworzenie klasyfikatora zespołowego z miękkim głosowaniem
    voting_clf = VotingClassifier(estimators=[('mlp', mlp), ('log_reg', log_reg), ('xgb', xgb)], voting='soft')

    # Trening klasyfikatora VotingClassifier na zbiorze treningowym
    voting_clf.fit(X_train, y_train)

    # Zwróć wytrenowany model oraz dane testowe
    return voting_clf, X_test, y_test
