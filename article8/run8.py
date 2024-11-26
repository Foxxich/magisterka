from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import numpy as np

def train_run8(X_train, y_train, X_test, y_test):
    """
    Trains a Voting Classifier using RandomForest and XGBoost.

    Parameters:
        X_train (np.ndarray): Training set features.
        y_train (list or np.ndarray): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (list or np.ndarray): Test set labels.

    Returns:
        ensemble_model: Trained Voting Classifier.
        X_test: Test set features.
        y_test: Test set labels.
    """
    # Convert sparse matrix to dense if necessary
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Define base classifiers
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Define ensemble model
    ensemble_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')

    # Train ensemble model
    ensemble_model.fit(X_train, y_train)

    return ensemble_model, X_test, y_test
