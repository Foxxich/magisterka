import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Funkcja do symulacji MC-Dropout na Random Forest
def monte_carlo_dropout_inference(model, data, num_samples=50):
    """
    Symuluje MC-Dropout poprzez wielokrotne próbkowanie predykcji z różnych podzbiorów drzew.
    """
    predictions = []
    n_trees = len(model.estimators_)  # Liczba drzew w lesie

    for _ in range(num_samples):
        # Wybierz losowy podzbiór drzew
        sampled_trees = np.random.choice(model.estimators_, size=int(n_trees * 0.8), replace=False)
        # Oblicz predykcje z wybranego podzbioru drzew
        tree_preds = np.array([tree.predict_proba(data) for tree in sampled_trees])
        # Uśrednij predykcje
        mean_tree_preds = tree_preds.mean(axis=0)
        predictions.append(mean_tree_preds)

    predictions = np.array(predictions)
    mean_preds = predictions.mean(axis=0)
    uncertainty = predictions.var(axis=0)
    return mean_preds, uncertainty

# Funkcja do obliczania prawdopodobieństw meta-atrybutów
def compute_meta_attribute_probabilities(df, attribute_col, label_col):
    attribute_probs = defaultdict(lambda: {'real': 0, 'fake': 0})

    for _, row in df.iterrows():
        attr_value = row[attribute_col]
        label = row[label_col]

        if label == 1:  # Wiadomość "prawdziwa"
            attribute_probs[attr_value]['real'] += 1
        elif label == 0:  # Wiadomość "fałszywa"
            attribute_probs[attr_value]['fake'] += 1

    for attr_value, counts in attribute_probs.items():
        total = counts['real'] + counts['fake']
        attribute_probs[attr_value]['real'] /= total
        attribute_probs[attr_value]['fake'] /= total

    return attribute_probs

# Heurystyczna obróbka post-procesowa
def heuristic_post_processing(predictions, attribute_probs, attributes, threshold=0.9):
    final_predictions = []
    for i, pred in enumerate(predictions):
        attr_value = attributes[i]
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

def metoda7(X_train, y_train, X_test, y_test):
    """
    Trenuje Random Forest z symulowanym MC-Dropout i heurystyczną obróbką post-procesową.

    Zwraca:
        model: Wytrenowany RandomForestClassifier.
        X_test: Oryginalne cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """

    # Konwersja danych wejściowych do tablic NumPy
    train_embeddings = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.array(X_train)
    test_embeddings = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.array(X_test)
    train_labels = y_train.to_numpy() if isinstance(y_train, pd.Series) else np.array(y_train)

    # Inicjalizacja i trening modelu Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    print("Debug: Trening modelu Random Forest...")
    model.fit(train_embeddings, train_labels)

    # Monte Carlo Dropout
    mean_preds, uncertainty = monte_carlo_dropout_inference(model, test_embeddings, num_samples=50)

    # Predykcja etykiet
    pred_labels = np.argmax(mean_preds, axis=1)

    # Przygotowanie danych dla heurystyk (uproszczone, używając 'unknown' dla źródła)
    df = pd.DataFrame({"label": y_test, "source": ["unknown"] * len(y_test) })

    # Obliczanie prawdopodobieństw atrybutów
    attribute_probs = compute_meta_attribute_probabilities(df, "source", "label")

    # Heurystyczna obróbka post-procesowa
    final_predictions = heuristic_post_processing(pred_labels, attribute_probs, df["source"].to_list(), threshold=0.9)

    # Ocena modelu (opcjonalnie - dodane dla kompletności)
    print("\nClassification Report:")
    print(classification_report(y_test, final_predictions))

    return model, X_test, y_test