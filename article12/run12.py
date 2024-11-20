from common import load_and_preprocess_data, vectorize_data, split_data, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import numpy as np


def train_run12(X_embeddings, X, y):
    """
    Trains and evaluates Random Forest and CatBoost classifiers.

    Args:
        X_embeddings (array-like): Precomputed embeddings or vectorized features.
        X (array-like): Original input data (used for compatibility).
        y (array-like): Target labels.

    Returns:
        dict: A dictionary containing trained models and their corresponding test datasets.
    """
    # Ensure the input embeddings are 2D
    if X_embeddings is None or len(X_embeddings.shape) != 2:
        raise ValueError("X_embeddings must be a 2-dimensional array.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

    # Ensure that the splits are properly converted to arrays if needed
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

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
