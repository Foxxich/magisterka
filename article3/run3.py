from common import load_and_preprocess_data, split_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
import numpy as np

def train_run3():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Ekstrakcja cech językowych (tymczasowy zamiennik dla specyficznych cech WELFake)
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    # Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_vec, y, test_size=0.3)

    # Definicja klasyfikatorów
    svm = SVC(probability=True, C=1.0, kernel='linear', gamma='scale')
    nb = MultinomialNB(alpha=0.5)
    dt = DecisionTreeClassifier(max_depth=5)
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=50, random_state=42)
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Klasyfikator głosowania pierwszego poziomu (Podzbiór cech językowych)
    voting_clf_1 = VotingClassifier(estimators=[
        ('svm', svm),
        ('nb', nb),
        ('dt', dt)
    ], voting='soft')

    # Trenowanie klasyfikatora głosowania pierwszego poziomu
    voting_clf_1.fit(X_train, y_train)
    
    # Predykcje pierwszego poziomu jako meta-cechy
    predictions_1_train = voting_clf_1.predict_proba(X_train)[:, 1]  # Dla trenowania drugiego poziomu
    predictions_1_test = voting_clf_1.predict_proba(X_test)[:, 1]   # Dla testowania drugiego poziomu

    # Połączenie predykcji pierwszego poziomu z cechami wejściowymi dla drugiego poziomu
    X_train_meta = np.column_stack((X_train.toarray(), predictions_1_train))
    X_test_meta = np.column_stack((X_test.toarray(), predictions_1_test))

    # Trenowanie finalnego klasyfikatora głosowania z Bagging i AdaBoost
    voting_clf_2 = VotingClassifier(estimators=[
        ('bagging', bagging),
        ('adaboost', adaboost)
    ], voting='soft')
    voting_clf_2.fit(X_train_meta, y_train)

    return voting_clf_2, X_test_meta, y_test
