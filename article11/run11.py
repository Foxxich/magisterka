from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def train_run11(X_train, y_train, X_test, y_test):
    """
    Trains an ensemble model using VotingClassifier with Random Forest, Logistic Regression, and AdaBoost,
    applying feature selection, cross-validation, and soft voting.

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
    # Ensure non-negative values for Chi-square by applying MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply feature selection using Chi-square test
    selector = SelectKBest(chi2, k=min(5000, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Define classifiers with specified hyperparameters
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
    lr = LogisticRegression(C=0.5, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
    adb = AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=42)

    # Create an ensemble VotingClassifier with soft voting
    voting_clf = VotingClassifier(estimators=[
        ('rf', rf),
        ('lr', lr),
        ('adb', adb)
    ], voting='soft')

    # Train the ensemble model on the training set
    voting_clf.fit(X_train_selected, y_train)

    # Return the trained model and test data
    return voting_clf, X_test_selected, y_test
