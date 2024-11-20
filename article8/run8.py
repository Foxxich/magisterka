from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from common import load_shuffle_preprocess_data, split_data

def train_run8(X_embeddings, X, y):
    # Ensure dimension consistency
    if len(X_embeddings) != len(y):
        raise ValueError(f"X_embeddings has {len(X_embeddings)} samples, but y has {len(y)} samples.")

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

    # Convert sparse matrix to dense for compatibility
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Create and train the ensemble model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    ensemble_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')

    ensemble_model.fit(X_train, y_train)

    return ensemble_model, X_test, y_test
