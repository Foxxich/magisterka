from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from common import vectorize_data, split_data

def train_run10(X_embeddings=None, X=None, y=None):
    """
    Trains a soft-voting ensemble model and evaluates it.
    
    Args:
        X_embeddings (np.ndarray or None): Precomputed embeddings or None for TF-IDF.
        X (list): Text data if embeddings need to be generated.
        y (list): Target labels.
        
    Returns:
        voting_clf: Trained VotingClassifier model.
        X_test: Test set features (embeddings or TF-IDF).
        y_test: Test set labels.
    """
    if X_embeddings is None:
        # If no embeddings provided, vectorize text data using TF-IDF
        X_tfidf, _ = vectorize_data(X, max_features=5000)
        X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.2)
    else:
        # Use precomputed embeddings
        X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

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
