from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler


def metoda11(X_train, y_train, X_test, y_test):
    """
    Trenuje model zespołowy z wykorzystaniem VotingClassifier (Random Forest, Logistic Regression i AdaBoost),
    stosując selekcję cech, normalizację i miękkie głosowanie.

    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (lista lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (lista lub np.ndarray): Etykiety zbioru testowego.

    Zwraca:
        voting_clf: Wytrenowany model VotingClassifier.
        X_test: Przetworzone cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Zapewnij nieujemne wartości do testu Chi-kwadrat przez zastosowanie MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Selekcja cech za pomocą testu Chi-kwadrat
    selector = SelectKBest(chi2, k=min(5000, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Definicja klasyfikatorów z określonymi hiperparametrami
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
    lr = LogisticRegression(C=0.5, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
    adb = AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=42)

    # Tworzenie zespołowego VotingClassifier z miękkim głosowaniem
    voting_clf = VotingClassifier(estimators=[
        ('rf', rf),
        ('lr', lr),
        ('adb', adb)
    ], voting='soft')

    # Trening modelu zespołowego na zbiorze treningowym
    voting_clf.fit(X_train_selected, y_train)

    # Zwróć wytrenowany model oraz przetworzone dane testowe
    return voting_clf, X_test_selected, y_test
