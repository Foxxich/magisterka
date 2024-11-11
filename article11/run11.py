from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from common import load_and_preprocess_data, vectorize_data, split_data

def train_run11():
    """
    Trenuj model zespołowy używając VotingClassifier z Random Forest, Logistic Regression i AdaBoost,
    stosując selekcję cech, walidację krzyżową i strategię głosowania miękkiego.
    """
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Wektoryzuj dane tekstowe
    X_tfidf, _ = vectorize_data(X, max_features=10000)

    # Zastosuj selekcję cech za pomocą testu Chi-kwadrat
    selector = SelectKBest(chi2, k=5000)
    X_tfidf = selector.fit_transform(X_tfidf, y)

    # Podziel dane na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)

    # Zdefiniuj klasyfikatory z określonymi hiperparametrami
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
    lr = LogisticRegression(C=0.5, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
    adb = AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=42)

    # VotingClassifier zespołowy z głosowaniem miękkim
    voting_clf = VotingClassifier(estimators=[
        ('rf', rf),
        ('lr', lr),
        ('adb', adb)
    ], voting='soft')
    
    # Wytrenuj model zespołowy na pełnym zbiorze treningowym
    voting_clf.fit(X_train, y_train)
    
    # Zwróć model, dane testowe i etykiety
    return voting_clf, X_test, y_test