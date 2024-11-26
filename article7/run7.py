from common import load_and_preprocess_data, split_data, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from collections import defaultdict

def monte_carlo_dropout_inference(model, X_test, num_samples=50):
    """Simulates Monte Carlo Dropout for RandomForest by making multiple predictions."""
    predictions = np.array([model.predict_proba(X_test) for _ in range(num_samples)])
    mean_preds = predictions.mean(axis=0)
    uncertainty = predictions.var(axis=0)
    return mean_preds, uncertainty

def compute_meta_attribute_probabilities(df, attribute_col, label_col):
    """Calculates the probability of an attribute indicating real or fake news."""
    attribute_probs = defaultdict(lambda: {'real': 0, 'fake': 0})

    for _, row in df.iterrows():
        attr_value = row[attribute_col]
        label = row[label_col]

        if label == 1:  # 'real' news
            attribute_probs[attr_value]['real'] += 1
        elif label == 0:  # 'fake' news
            attribute_probs[attr_value]['fake'] += 1

    # Normalize probabilities
    for attr_value, counts in attribute_probs.items():
        total = counts['real'] + counts['fake']
        attribute_probs[attr_value]['real'] /= total
        attribute_probs[attr_value]['fake'] /= total

    return attribute_probs

def heuristic_post_processing(predictions, attribute_probs, attributes, threshold=0.9):
    """Applies heuristic post-processing based on attribute probabilities."""
    attributes = list(attributes)  # Ensure attributes is a list
    final_predictions = []
    for i, pred in enumerate(predictions):
        attr_value = attributes[i]  # Get the attribute value by index
        if attr_value in attribute_probs:
            real_prob = attribute_probs[attr_value]['real']
            fake_prob = attribute_probs[attr_value]['fake']
            if real_prob > threshold and real_prob > fake_prob:
                final_predictions.append(1)
            elif fake_prob > threshold and fake_prob > real_prob:
                final_predictions.append(0)
            else:
                final_predictions.append(pred)
        else:
            final_predictions.append(pred)
    return final_predictions

def train_run7(X_embeddings=None, X=None, y=None):
    """
    Trains a RandomForest model and applies heuristic post-processing.

    Args:
        X_embeddings (np.ndarray or None): Precomputed embeddings.
        X (list or pd.Series): Raw text data (if embeddings are not provided).
        y (list or pd.Series): Target labels.

    Returns:
        rf_classifier: Trained RandomForest model.
        X_test: Original or preprocessed test features.
        y_test: Test set labels.
    """
    # If embeddings are provided, use them; otherwise, preprocess text data
    if X_embeddings is None:
        # Load and preprocess data
        X, y = load_and_preprocess_data()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        # Vectorize text data using TF-IDF
        print("Debug: Applying TF-IDF vectorization...")
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, max_df=0.7, stop_words="english")
        X_train = tfidf_vectorizer.fit_transform(X_train)
        X_test = tfidf_vectorizer.transform(X_test)
    else:
        # Use precomputed embeddings
        X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

    # Train RandomForest model
    print("Debug: Training RandomForest model...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Monte Carlo Dropout
    print("Debug: Applying Monte Carlo Dropout inference...")
    mean_preds, uncertainty = monte_carlo_dropout_inference(rf_classifier, X_test, num_samples=50)

    # Predict labels
    pred_labels = np.argmax(mean_preds, axis=1)

    # Handle meta-attributes for heuristic post-processing
    print("Debug: Creating DataFrame for meta-attribute analysis...")
    df = pd.DataFrame({"text": X if X_embeddings is None else [""] * len(X_test), "label": y_test})
    df["source"] = ["unknown" for _ in range(len(df))]  # Replace with actual meta-attribute column

    # Compute attribute probabilities
    print("Debug: Computing meta-attribute probabilities...")
    attribute_probs = compute_meta_attribute_probabilities(df, "source", "label")

    # Apply heuristic post-processing
    print("Debug: Applying heuristic post-processing...")
    final_predictions = heuristic_post_processing(
        pred_labels,
        attribute_probs,
        df["source"].reset_index(drop=True),  # Ensure indices match
        threshold=0.9
    )

    # Debug final predictions
    print(f"Debug: Final predictions shape: {len(final_predictions)}")

    return rf_classifier, X_test, y_test
