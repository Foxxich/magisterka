from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
import numpy as np
from common import vectorize_data, split_data

def train_run3(X_embeddings=None, X=None, y=None):
    """
    Trains a two-level ensemble model using feature-based classifiers and meta-classifiers.

    Parameters:
        X_embeddings (np.ndarray or None): Precomputed embeddings or None for raw text.
        X (list): Input text data for vectorization.
        y (list): Target labels.

    Returns:
        voting_clf_2: Trained second-level ensemble model.
        X_test_meta: Test set meta-features.
        y_test: Test set labels.
    """
    # If embeddings are not provided, vectorize the text data
    if X_embeddings is None:
        X_vec, _ = vectorize_data(X, max_features=5000, ngram_range=(1, 2))
    else:
        X_vec = X_embeddings

    # Ensure non-negative values for MultinomialNB
    if hasattr(X_vec, "toarray"):
        X_vec = X_vec.toarray()
    X_vec = np.abs(X_vec)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_vec, y, test_size=0.3)

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
    predictions_1_train = voting_clf_1.predict_proba(X_train)[:, 1]
    predictions_1_test = voting_clf_1.predict_proba(X_test)[:, 1]

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
