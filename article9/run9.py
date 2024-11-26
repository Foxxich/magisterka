from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
)
import numpy as np

def train_run9(X_train, y_train, X_test, y_test):
    """
    Trains various ensemble models and evaluates them.
    
    Parameters:
        X_train (np.ndarray): Training set features.
        y_train (list or np.ndarray): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (list or np.ndarray): Test set labels.
    
    Returns:
        voting: Best model (VotingClassifier).
        X_test: Test set features.
        y_test: Test set labels.
    """
    # Define base classifiers
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    svc = SVC(kernel='linear', probability=True, random_state=42)

    # Define ensemble methods
    bagging = BaggingClassifier(estimator=rf, n_estimators=50, random_state=42)
    boosting = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    voting = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('svc', svc)], voting='soft')

    # Train models and evaluate
    models = {
        "Random Forest": rf,
        "Bagging": bagging,
        "Boosting (AdaBoost)": boosting,
        "Voting Classifier": voting
    }

    results = {}
    for model_name, model in models.items():
        print(f"Trenowanie {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
        results[model_name] = {
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
            "Classification Report": classification_report(y_test, y_pred, output_dict=True)
        }
        print(f"Dokładność dla {model_name}: {accuracy}")
        if roc_auc is not None:
            print(f"ROC-AUC dla {model_name}: {roc_auc}")
        print(classification_report(y_test, y_pred))

    # Return the best model (voting classifier) and test data
    return voting, X_test, y_test
