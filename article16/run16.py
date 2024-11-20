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


def train_run16(X_embeddings=None, X=None, y=None, max_features=5000, test_size=0.2):
    """
    Trains a stacking ensemble model with meta-classifier using logistic regression.

    Parameters:
        X_embeddings (np.ndarray): Precomputed embeddings or vectorized features. Overrides `X` if provided.
        X (pd.Series): Input text data. Used only if `X_embeddings` is not provided.
        y (pd.Series): Target labels. Must be provided.
        max_features (int): Maximum number of features for TF-IDF vectorization (used only if `X_embeddings` is None).
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        meta_model: Trained meta-classifier (stacking).
        meta_features_test: Meta-features for the test set.
        y_test: Labels for the test set.
    """
    if y is None:
        raise ValueError("The target variable `y` must be provided.")

    # Use X_embeddings if provided, otherwise preprocess and vectorize `X`
    if X_embeddings is None:
        if X is None:
            raise ValueError("Either `X_embeddings` or `X` must be provided.")
        # Preprocess text data
        X = preprocess_text(X)
        # TF-IDF vectorization
        X_embeddings, _ = vectorize_data(X, max_features=max_features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=test_size)

    # Initialize base models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svc_model = SVC(probability=True, kernel="linear", random_state=42)

    # Generate meta-features for training using cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_preds_train = cross_val_predict(rf_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    gb_preds_train = cross_val_predict(gb_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]
    svc_preds_train = cross_val_predict(svc_model, X_train, y_train, method="predict_proba", cv=skf)[:, 1]

    meta_features_train = np.column_stack((rf_preds_train, gb_preds_train, svc_preds_train))

    # Train meta-classifier using logistic regression
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(meta_features_train, y_train)

    # Fit base models on the full training data
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)

    # Generate meta-features for testing
    rf_preds_test = rf_model.predict_proba(X_test)[:, 1]
    gb_preds_test = gb_model.predict_proba(X_test)[:, 1]
    svc_preds_test = svc_model.predict_proba(X_test)[:, 1]
    meta_features_test = np.column_stack((rf_preds_test, gb_preds_test, svc_preds_test))

    # Evaluate and print the performance
    y_pred = meta_model.predict(meta_features_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return meta_model, meta_features_test, y_test
