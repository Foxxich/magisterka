from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
import numpy as np

def train_run3(X_train, y_train, X_test, y_test):
    """
    Trains a two-level ensemble model using feature-based classifiers and meta-classifiers.
    
    Parameters:
        X_train (np.ndarray): Training set features.
        y_train (list or np.ndarray): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (list or np.ndarray): Test set labels.
    
    Returns:
        voting_clf_2: Trained second-level ensemble model.
        X_test_meta: Test set meta-features.
        y_test: Test set labels.
    """
    # Ensure non-negative values for MultinomialNB
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    X_train = np.abs(X_train)
    X_test = np.abs(X_test)

    # Define first-level classifiers
    svm = SVC(probability=True, C=1.0, kernel='linear', gamma='scale')
    nb = MultinomialNB(alpha=0.5)
    dt = DecisionTreeClassifier(max_depth=5)

    # First-level ensemble (Voting Classifier)
    voting_clf_1 = VotingClassifier(estimators=[
        ('svm', svm),
        ('nb', nb),
        ('dt', dt)
    ], voting='soft')

    # Train first-level classifier
    voting_clf_1.fit(X_train, y_train)

    # Generate first-level predictions as meta-features
    predictions_1_train = voting_clf_1.predict_proba(X_train)
    predictions_1_test = voting_clf_1.predict_proba(X_test)

    # Combine first-level predictions with original features
    X_train_meta = np.column_stack((X_train, predictions_1_train))
    X_test_meta = np.column_stack((X_test, predictions_1_test))

    # Define second-level classifiers
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=50, random_state=42)
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Second-level ensemble (Voting Classifier)
    voting_clf_2 = VotingClassifier(estimators=[
        ('bagging', bagging),
        ('adaboost', adaboost)
    ], voting='soft')

    # Train second-level classifier
    voting_clf_2.fit(X_train_meta, y_train)

    return voting_clf_2, X_test_meta, y_test
