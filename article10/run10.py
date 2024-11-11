from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from common import load_and_preprocess_data, vectorize_data, split_data

def train_run10():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Wektoryzuj dane tekstowe
    X_tfidf, _ = vectorize_data(X, max_features=5000)

    # Podziel dane na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)

    # Inicjalizacja klasyfikatorów z dostosowanymi hiperparametrami
    mlp = MLPClassifier(alpha=0.01, hidden_layer_sizes=(14,), max_iter=100, solver='lbfgs', random_state=0)
    log_reg = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=100, random_state=0)
    xgb = XGBClassifier(gamma=1, learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=100,
                        use_label_encoder=False, eval_metric='logloss')

    # Stwórz VotingClassifier z głosowaniem miękkim
    voting_clf = VotingClassifier(estimators=[('mlp', mlp), ('log_reg', log_reg), ('xgb', xgb)], voting='soft')

    # Wytrenuj VotingClassifier na pełnym zbiorze treningowym
    voting_clf.fit(X_train, y_train)

    # Zwróć wytrenowany model i dane testowe do oceny
    return voting_clf, X_test, y_test
