import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common import vectorize_data, split_data


def preprocess_text(X):
    """
    Custom text preprocessing: convert to lowercase, remove stopwords, and apply stemming.
    """
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    import re

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    def clean_text(text):
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\d+", "", text)  # Remove numbers
        words = text.lower().split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return " ".join(words)

    return X.apply(clean_text)

def train_run16(X_train, y_train, X_test, y_test):
    """
    Trains a stacking ensemble model with meta-classifier using logistic regression.

    Parameters:
        X_train (np.ndarray): Training set features.
        y_train (np.ndarray): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (np.ndarray): Test set labels.

    Returns:
        meta_model: Trained meta-classifier (stacking).
        meta_features_test: Meta-features for the test set.
        y_test: Labels for the test set.
    """
    # Initialize base models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svc_model = SVC(probability=True, kernel="linear", random_state=42)

    # Generate meta-features for training using cross-validation
    print("Generating meta-features for training...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_preds_train = cross_val_predict(rf_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    gb_preds_train = cross_val_predict(gb_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    svc_preds_train = cross_val_predict(svc_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]

    meta_features_train = np.column_stack((rf_preds_train, gb_preds_train, svc_preds_train))

    # Train meta-classifier using logistic regression
    print("Training meta-classifier...")
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(meta_features_train, y_train)

    # Fit base models on the full training data
    print("Training base models...")
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)

    # Generate meta-features for testing
    print("Generating meta-features for testing...")
    rf_preds_test = rf_model.predict_proba(X_test)[:, 1]
    gb_preds_test = gb_model.predict_proba(X_test)[:, 1]
    svc_preds_test = svc_model.predict_proba(X_test)[:, 1]
    meta_features_test = np.column_stack((rf_preds_test, gb_preds_test, svc_preds_test))

    # Evaluate and print the performance
    print("Evaluating model performance...")
    y_pred = meta_model.predict(meta_features_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return meta_model, meta_features_test, y_test
