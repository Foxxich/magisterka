from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
import numpy as np

def metoda3(X_train, y_train, X_test, y_test):
    """
    Trenuje dwupoziomowy model zespołowy wykorzystujący klasyfikatory cechowe i meta-klasyfikatory.
    
    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (list lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (list lub np.ndarray): Etykiety zbioru testowego.
    
    Zwraca:
        voting_clf_2: Wytrenowany model zespołowy drugiego poziomu.
        X_test_meta: Meta-cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Zapewnij nieujemne wartości dla MultinomialNB
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    X_train = np.abs(X_train)
    X_test = np.abs(X_test)

    # Definicja klasyfikatorów pierwszego poziomu
    svm = SVC(probability=True, C=1.0, kernel='linear', gamma='scale')
    nb = MultinomialNB(alpha=0.5)
    dt = DecisionTreeClassifier(max_depth=5)

    # Zespół pierwszego poziomu (Voting Classifier)
    voting_clf_1 = VotingClassifier(estimators=[
        ('svm', svm),
        ('nb', nb),
        ('dt', dt)
    ], voting='soft')

    # Trenuj klasyfikator pierwszego poziomu
    voting_clf_1.fit(X_train, y_train)

    # Generowanie predykcji pierwszego poziomu jako meta-cech
    predictions_1_train = voting_clf_1.predict_proba(X_train)
    predictions_1_test = voting_clf_1.predict_proba(X_test)

    # Łączenie predykcji pierwszego poziomu z oryginalnymi cechami
    X_train_meta = np.column_stack((X_train, predictions_1_train))
    X_test_meta = np.column_stack((X_test, predictions_1_test))

    # Definicja klasyfikatorów drugiego poziomu
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=50, random_state=42)
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Zespół drugiego poziomu (Voting Classifier)
    voting_clf_2 = VotingClassifier(estimators=[
        ('bagging', bagging),
        ('adaboost', adaboost)
    ], voting='soft')

    # Trenuj klasyfikator drugiego poziomu
    voting_clf_2.fit(X_train_meta, y_train)

    return voting_clf_2, X_test_meta, y_test
