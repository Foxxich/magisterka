from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import numpy as np
from common import load_and_preprocess_data, split_data


def train_run13(X_embeddings, X, y):
    """
    Trains an XGBoost classifier with SMOTETomek for handling class imbalance.

    Args:
        X_embeddings (array-like): Precomputed embeddings or vectorized features.
        X (array-like): Original input data (used for compatibility).
        y (array-like): Target labels.

    Returns:
        XGBClassifier: Trained XGBoost model.
        np.ndarray: Test features.
        np.ndarray: Test labels.
    """
    # Ensure X_embeddings is 2D
    if X_embeddings is None or len(X_embeddings.shape) != 2:
        raise ValueError("X_embeddings must be a 2-dimensional array.")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.3, random_state=42)

    # Ensure data is in the correct format
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Handle class imbalance using SMOTETomek
    smotetomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train, y_train)

    # Define and train the XGBoost model
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train_resampled) / np.sum(y_train_resampled == 1),
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        random_state=42
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Return the trained model and test data
    return xgb_model, X_test, y_test
