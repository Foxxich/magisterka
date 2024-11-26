from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import numpy as np

def train_run12(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates Random Forest and CatBoost classifiers.

    Parameters:
        X_train (array-like): Training set features.
        y_train (array-like): Training set labels.
        X_test (array-like): Test set features.
        y_test (array-like): Test set labels.

    Returns:
        dict: A dictionary containing trained models and their corresponding test datasets.
    """
    # Ensure the input embeddings are 2D
    if len(X_train.shape) != 2 or len(X_test.shape) != 2:
        raise ValueError("Input features must be 2-dimensional arrays.")

    # Train Random Forest (Bagging)
    print("Training Random Forest...")
    rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Train CatBoost (Boosting)
    print("Training CatBoost...")
    catboost_classifier = CatBoostClassifier(
        iterations=200,
        learning_rate=0.01,
        eval_metric='Accuracy',
        early_stopping_rounds=20,
        use_best_model=True,
        verbose=50
    )
    catboost_classifier.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

    # Return both models with their test data for evaluation
    return {
        "RandomForest": (rf_classifier, X_test, y_test),
        "CatBoost": (catboost_classifier, X_test, y_test)
    }
