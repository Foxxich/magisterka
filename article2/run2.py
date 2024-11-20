from common import vectorize_data, split_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train_run2(X_embeddings=None, X=None, y=None):
    """
    Trains a model using GradientBoosting and MLP on separate feature subsets 
    and combines them with Logistic Regression.
    
    Parameters:
        X_embeddings (np.ndarray or None): Precomputed embeddings or None for TF-IDF.
        X (list): Input text data for vectorization.
        y (list): Target labels.

    Returns:
        logistic_reg: Trained Logistic Regression model.
        combined_preds: Combined predictions (features) used for meta-model.
        y_test: Test set labels.
    """
    # If embeddings are not provided, vectorize the text data using TF-IDF
    if X_embeddings is None:
        X_vec, _ = vectorize_data(X, max_features=5000)
    else:
        X_vec = X_embeddings

    # Convert to DataFrame for feature selection
    X_df = pd.DataFrame(X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec)

    # Split features into demographic and social behavior subsets
    num_features = X_df.shape[1]
    if num_features < 6:
        raise ValueError("Insufficient features to split into demographic and social behavior subsets.")
    demographic_features = X_df.iloc[:, :5]  # First 5 columns for demographic features
    social_behavior_features = X_df.iloc[:, 5:]  # Remaining columns for social behavior features

    # Split the data into training and test sets
    demo_train, demo_test, y_train, y_test = split_data(demographic_features, y, test_size=0.35)
    soc_train, soc_test, _, _ = split_data(social_behavior_features, y, test_size=0.35)

    # Train GradientBoostingClassifier on social behavior features
    boosted_tree = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    boosted_tree.fit(soc_train, y_train)
    boosted_tree_preds = boosted_tree.predict_proba(soc_test)[:, 1]

    # Train MLPClassifier on demographic features
    neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    neural_net.fit(demo_train, y_train)
    neural_net_preds = neural_net.predict_proba(demo_test)[:, 1]

    # Combine predictions into a single DataFrame
    combined_preds = pd.DataFrame({
        'boosted_tree': boosted_tree_preds,
        'neural_net': neural_net_preds
    })

    # Train Logistic Regression on combined predictions
    logistic_reg = LogisticRegression()
    logistic_reg.fit(combined_preds, y_test)

    return logistic_reg, combined_preds, y_test
