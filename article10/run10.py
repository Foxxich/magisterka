from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np

def train_run10(X_train, y_train, X_test, y_test):
    """
    Trains a soft-voting ensemble model and evaluates it.

    Parameters:
        X_train (np.ndarray): Training set features.
        y_train (list or np.ndarray): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (list or np.ndarray): Test set labels.

    Returns:
        voting_clf: Trained VotingClassifier model.
        X_test: Test set features.
        y_test: Test set labels.
    """
    # Initialize classifiers with tuned hyperparameters
    mlp = MLPClassifier(alpha=0.01, hidden_layer_sizes=(14,), max_iter=100, solver='lbfgs', random_state=0)
    log_reg = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=100, random_state=0)
    xgb = XGBClassifier(
        gamma=1, learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=100,
        use_label_encoder=False, eval_metric='logloss'
    )

    # Create a VotingClassifier with soft voting
    voting_clf = VotingClassifier(estimators=[('mlp', mlp), ('log_reg', log_reg), ('xgb', xgb)], voting='soft')

    # Train the VotingClassifier on the training set
    voting_clf.fit(X_train, y_train)

    # Return the trained model and test data
    return voting_clf, X_test, y_test
