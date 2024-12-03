from common import load_and_preprocess_data, split_data, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from collections import defaultdict

def monte_carlo_dropout_inference(model, X_test, num_samples=50):
    """Symuluje Monte Carlo Dropout dla RandomForest poprzez wielokrotne predykcje."""
    predictions = np.array([model.predict_proba(X_test) for _ in range(num_samples)])
    mean_preds = predictions.mean(axis=0)
    uncertainty = predictions.var(axis=0)
    return mean_preds, uncertainty

def compute_meta_attribute_probabilities(df, attribute_col, label_col):
    """Oblicza prawdopodobieństwo, że atrybut wskazuje na prawdziwe lub fałszywe wiadomości."""
    attribute_probs = defaultdict(lambda: {'real': 0, 'fake': 0})

    for _, row in df.iterrows():
        attr_value = row[attribute_col]
        label = row[label_col]

        if label == 1:  # Wiadomość "prawdziwa"
            attribute_probs[attr_value]['real'] += 1
        elif label == 0:  # Wiadomość "fałszywa"
            attribute_probs[attr_value]['fake'] += 1

    # Normalizacja prawdopodobieństw
    for attr_value, counts in attribute_probs.items():
        total = counts['real'] + counts['fake']
        attribute_probs[attr_value]['real'] /= total
        attribute_probs[attr_value]['fake'] /= total

    return attribute_probs

def heuristic_post_processing(predictions, attribute_probs, attributes, threshold=0.9):
    """Stosuje heurystyczną obróbkę post-procesową na podstawie prawdopodobieństw atrybutów."""
    attributes = list(attributes)  # Upewnij się, że attributes jest listą
    final_predictions = []
    for i, pred in enumerate(predictions):
        attr_value = attributes[i]  # Pobierz wartość atrybutu według indeksu
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
    Trenuje model RandomForest i stosuje heurystyczną obróbkę post-procesową.

    Parametry:
        X_embeddings (np.ndarray lub None): Wstępnie obliczone osadzenia.
        X (lista lub pd.Series): Surowe dane tekstowe (jeśli osadzenia nie są podane).
        y (lista lub pd.Series): Etykiety docelowe.

    Zwraca:
        rf_classifier: Wytrenowany model RandomForest.
        X_test: Oryginalne lub przetworzone cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Jeśli osadzenia są podane, użyj ich; w przeciwnym razie przetwórz dane tekstowe
    if X_embeddings is None:
        # Wczytaj i przetwórz dane
        X, y = load_and_preprocess_data()

        # Podziel dane na zbiory treningowy i testowy
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        # Wektoryzacja danych tekstowych za pomocą TF-IDF
        print("Debug: Stosowanie wektoryzacji TF-IDF...")
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, max_df=0.7, stop_words="english")
        X_train = tfidf_vectorizer.fit_transform(X_train)
        X_test = tfidf_vectorizer.transform(X_test)
    else:
        # Użyj wstępnie obliczonych osadzeń
        X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

    # Trenuj model RandomForest
    print("Debug: Trening modelu RandomForest...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Monte Carlo Dropout
    print("Debug: Stosowanie wnioskowania Monte Carlo Dropout...")
    mean_preds, uncertainty = monte_carlo_dropout_inference(rf_classifier, X_test, num_samples=50)

    # Predykcja etykiet
    pred_labels = np.argmax(mean_preds, axis=1)

    # Obsługa meta-atrybutów dla heurystycznej obróbki post-procesowej
    print("Debug: Tworzenie DataFrame do analizy meta-atrybutów...")
    df = pd.DataFrame({"text": X if X_embeddings is None else [""] * len(X_test), "label": y_test})
    df["source"] = ["unknown" for _ in range(len(df))]  # Zastąp rzeczywistą kolumną meta-atrybutów

    # Obliczanie prawdopodobieństw atrybutów
    print("Debug: Obliczanie prawdopodobieństw meta-atrybutów...")
    attribute_probs = compute_meta_attribute_probabilities(df, "source", "label")

    # Stosowanie heurystycznej obróbki post-procesowej
    print("Debug: Stosowanie heurystycznej obróbki post-procesowej...")
    final_predictions = heuristic_post_processing(
        pred_labels,
        attribute_probs,
        df["source"].reset_index(drop=True),  # Upewnij się, że indeksy pasują
        threshold=0.9
    )

    # Debug finalnych predykcji
    print(f"Debug: Kształt finalnych predykcji: {len(final_predictions)}")

    return rf_classifier, X_test, y_test
