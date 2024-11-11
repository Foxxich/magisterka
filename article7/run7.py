from common import load_and_preprocess_data, split_data, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from collections import defaultdict

def monte_carlo_dropout_inference(model, X_test, num_samples=50):
    """Symuluje Monte Carlo Dropout dla RandomForest przez wielokrotne predykcje."""
    predictions = np.array([model.predict_proba(X_test) for _ in range(num_samples)])
    mean_preds = predictions.mean(axis=0)
    uncertainty = predictions.var(axis=0)
    return mean_preds, uncertainty

def compute_meta_attribute_probabilities(df, attribute_col, label_col):
    """Oblicza prawdopodobieństwo atrybutu wskazującego na prawdziwe lub fałszywe wiadomości."""
    attribute_probs = defaultdict(lambda: {'real': 0, 'fake': 0})

    for _, row in df.iterrows():
        attr_value = row[attribute_col]
        label = row[label_col]

        if label == 1:  # 'prawdziwe' wiadomości
            attribute_probs[attr_value]['real'] += 1
        elif label == 0:  # 'fałszywe' wiadomości
            attribute_probs[attr_value]['fake'] += 1

    # Normalizacja prawdopodobieństw
    for attr_value, counts in attribute_probs.items():
        total = counts['real'] + counts['fake']
        attribute_probs[attr_value]['real'] /= total
        attribute_probs[attr_value]['fake'] /= total

    return attribute_probs

def heuristic_post_processing(predictions, attribute_probs, attributes, threshold=0.9):
    """Stosuje heurystyczne poprawki na podstawie prawdopodobieństw atrybutów."""
    attributes = list(attributes)  # Upewnij się, że attributes jest listą
    final_predictions = []
    for i, pred in enumerate(predictions):
        attr_value = attributes[i]  # Uzyskaj wartość atrybutu na podstawie indeksu
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

def train_run7():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()
    
    # Podziel dane na zbiory
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Wektoryzuj dane tekstowe przy użyciu TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, max_df=0.7, stop_words="english")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Trenuj model RandomForest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_tfidf, y_train)

    # Monte Carlo Dropout
    mean_preds, uncertainty = monte_carlo_dropout_inference(rf_classifier, X_test_tfidf, num_samples=50)

    # Predykcja etykiet
    pred_labels = np.argmax(mean_preds, axis=1)

    # Utwórz DataFrame z tekstem testowym i etykietami
    df = pd.DataFrame({"text": X_test, "label": y_test})
    df["source"] = ["unknown" for _ in range(len(df))]  # Zastąp rzeczywistą kolumną meta-atrybutu

    # Oblicz prawdopodobieństwa atrybutów
    attribute_probs = compute_meta_attribute_probabilities(df, "source", "label")

    # Zastosuj heurystyczne post-processing
    final_predictions = heuristic_post_processing(
        pred_labels, 
        attribute_probs, 
        df["source"].reset_index(drop=True),  # Upewnij się, że indeksy pasują
        threshold=0.9
    )

    return rf_classifier, X_test_tfidf, y_test, final_predictions