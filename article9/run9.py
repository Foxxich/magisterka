from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from common import vectorize_data, split_data
import numpy as np

def train_run9(X_embeddings=None, X=None, y=None, batch_size=32):
    """
    Trains various ensemble models and evaluates them.
    
    Args:
        X_embeddings (np.ndarray or None): Precomputed embeddings or None for TF-IDF.
        X (list): Text data if embeddings need to be generated.
        y (list): Target labels.
        batch_size (int): Batch size for embeddings (if applicable).
        
    Returns:
        voting: Best model (VotingClassifier).
        X_test: Test set features (embeddings or TF-IDF).
        y_test: Test set labels.
    """
    if X_embeddings is None:
        # Vectorize text data using TF-IDF if embeddings are not provided
        tfidf_vectorizer = vectorize_data(X, max_features=5000)[1]
        X_train, X_test, y_train, y_test = split_data(tfidf_vectorizer.transform(X), y, test_size=0.3)
    else:
        # Use precomputed embeddings
        X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.3)

    # Define base classifiers with hyperparameter tuning
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
        report = classification_report(y_test, y_pred, output_dict=True)
        results[model_name] = {
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
            "Classification Report": report
        }
        print(f"Dokładność dla {model_name}: {accuracy}")
        if roc_auc:
            print(f"ROC-AUC dla {model_name}: {roc_auc}")
        print(classification_report(y_test, y_pred))

    # Return the best model (voting classifier) and test data
    return voting, X_test, y_test
