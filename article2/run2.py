from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def train_run2(X_train, y_train, X_test, y_test):
    """
    Trains a model using GradientBoosting and MLP on separate feature subsets 
    and combines them with Logistic Regression.
    
    Parameters:
        X_train (np.ndarray or DataFrame): Training set features.
        y_train (list or np.ndarray): Training set labels.
        X_test (np.ndarray or DataFrame): Test set features.
        y_test (list or np.ndarray): Test set labels.
    
    Returns:
        logistic_reg: Trained Logistic Regression model.
        combined_preds_test: Combined predictions (features) used for meta-model (test set).
        y_test: Test set labels.
    """
    # Ensure input features are in DataFrame format
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    # Split features into demographic and social behavior subsets
    num_features = X_train_df.shape[1]
    if num_features < 6:
        raise ValueError("Insufficient features to split into demographic and social behavior subsets.")
    demographic_features_train = X_train_df.iloc[:, :5]
    social_behavior_features_train = X_train_df.iloc[:, 5:]
    demographic_features_test = X_test_df.iloc[:, :5]
    social_behavior_features_test = X_test_df.iloc[:, 5:]

    # Train GradientBoostingClassifier on social behavior features
    boosted_tree = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    boosted_tree.fit(social_behavior_features_train, y_train)
    boosted_tree_preds_test = boosted_tree.predict_proba(social_behavior_features_test)[:, 1]

    # Train MLPClassifier on demographic features
    neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    neural_net.fit(demographic_features_train, y_train)
    neural_net_preds_test = neural_net.predict_proba(demographic_features_test)[:, 1]

    # Combine predictions into a single DataFrame
    combined_preds_test = pd.DataFrame({
        'boosted_tree': boosted_tree_preds_test,
        'neural_net': neural_net_preds_test
    })

    # Train Logistic Regression on combined predictions
    logistic_reg = LogisticRegression()
    logistic_reg.fit(combined_preds_test, y_test)

    return logistic_reg, combined_preds_test, y_test
