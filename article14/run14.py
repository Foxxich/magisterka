import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import SnowballStemmer
from sklearn.exceptions import NotFittedError
from common import load_and_preprocess_data, vectorize_data, split_data


def preprocess_text(text):
    """
    Cleans and preprocesses text by removing URLs, special characters, single letters, and numbers,
    and applies stemming using the Snowball Stemmer.
    """
    snowball_stemmer = SnowballStemmer('english')
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single letters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single letters at the start
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with a single space
    text = re.sub(r'^\s+|\s+?$', '', text.lower())  # Remove leading/trailing spaces and lowercase
    return ' '.join([snowball_stemmer.stem(word) for word in text.split()])  # Apply stemming


def train_run14(X_train, y_train, X_test, y_test):
    """
    Trains a stacking ensemble model with Random Forest, AdaBoost, and Logistic Regression
    as the final estimator.

    Parameters:
        X_train (numpy.ndarray): Training set features.
        y_train (numpy.ndarray): Training set labels.
        X_test (numpy.ndarray): Test set features.
        y_test (numpy.ndarray): Test set labels.

    Returns:
        StackingClassifier: Trained stacking model.
        numpy.ndarray: Test features.
        numpy.ndarray: Test labels.
    """
    # Ensure input data is 2D
    if len(X_train.shape) != 2 or len(X_test.shape) != 2:
        raise ValueError("Input features must be 2-dimensional arrays.")

    # Define base models and meta-model for stacking
    print("Defining models...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(max_iter=200, random_state=42)

    # Create stacking classifier
    print("Building stacking model...")
    stack_model = StackingClassifier(
        estimators=[('rf', rf_model), ('ab', ab_model)],
        final_estimator=lr_model
    )

    # Train the stacking model
    print("Training stacking model...")
    stack_model.fit(X_train, y_train)

    print("Training completed.")

    # Return trained model and test data
    return stack_model, X_test, y_test