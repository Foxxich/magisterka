from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import numpy as np

def train_run13(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost classifier with SMOTETomek for handling class imbalance.

    Parameters:
        X_train (np.ndarray): Training set features.
        y_train (array-like): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (array-like): Test set labels.

    Returns:
        XGBClassifier: Trained XGBoost model.
        np.ndarray: Test features.
        np.ndarray: Test labels.
    """
    # Ensure data is in the correct format
    if len(X_train.shape) != 2 or len(X_test.shape) != 2:
        raise ValueError("Input features must be 2-dimensional arrays.")

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
