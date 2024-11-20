from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from common import vectorize_data, split_data

def train_run11(X_embeddings=None, X=None, y=None):
    """
    Trains an ensemble model using VotingClassifier with Random Forest, Logistic Regression, and AdaBoost,
    applying feature selection, cross-validation, and soft voting.
    
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
        # If no embeddings are provided, vectorize the text data using TF-IDF
        X_tfidf, _ = vectorize_data(X, max_features=5000)
    else:
        # Use precomputed embeddings
        X_tfidf = X_embeddings

    # Ensure non-negative values for Chi-square by applying MinMaxScaler
    scaler = MinMaxScaler()
    X_tfidf = scaler.fit_transform(X_tfidf)

    # Apply feature selection using Chi-square test
    selector = SelectKBest(chi2, k=min(5000, X_tfidf.shape[1]))
    X_tfidf = selector.fit_transform(X_tfidf, y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.2)

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
    voting_clf.fit(X_train, y_train)

    # Return the trained model and test data
    return voting_clf, X_test, y_test
