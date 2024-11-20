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


def train_run14(X_embeddings=None, X=None, y=None):
    """
    Trains a stacking ensemble model with Random Forest, AdaBoost, and Logistic Regression
    as the final estimator. Handles both precomputed embeddings or raw text data.

    Args:
        X_embeddings (numpy.ndarray or sparse matrix): Precomputed feature embeddings.
        X (pandas.Series): Raw text data for vectorization (if embeddings are not provided).
        y (numpy.ndarray or pandas.Series): Target labels.

    Returns:
        StackingClassifier: Trained stacking model.
        numpy.ndarray: Test features.
        numpy.ndarray: Test labels.
    """
    # Handle raw text data if embeddings are not provided
    if X_embeddings is None:
        if X is None or y is None:
            raise ValueError("Both `X` and `y` must be provided if `X_embeddings` is not supplied.")
        
        # Apply additional preprocessing to text data
        print("Preprocessing text data...")
        X = X.apply(preprocess_text)

        # Vectorize text data using TF-IDF
        print("Vectorizing text data...")
        X_embeddings, _ = vectorize_data(X, max_features=5000)

    # Split data into training and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

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
    try:
        stack_model.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during training: {e}")
        raise

    print("Training completed.")

    # Return trained model and test data
    return stack_model, X_test, y_test
