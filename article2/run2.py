from common import load_and_preprocess_data, vectorize_data, split_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train_run2():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()
    X_vec, _ = vectorize_data(X, max_features=5000)

    # Konwersja X na DataFrame (rozwiązanie problemu z indeksowaniem)
    X = pd.DataFrame(X_vec.toarray())

    # Podziel cechy na podzbiory demograficzne i dotyczące zachowań społecznych
    demographic_features = X.iloc[:, :5]  # Pierwsze 5 kolumn jako demograficzne
    social_behavior_features = X.iloc[:, 5:]  # Pozostałe kolumny jako zachowania społeczne

    # Podziel dane na zbiory treningowe i testowe
    demo_train, demo_test, y_train, y_test = split_data(demographic_features, y, test_size=0.35)
    soc_train, soc_test, _, _ = split_data(social_behavior_features, y, test_size=0.35)

    # Trenowanie Boosted Decision Tree
    boosted_tree = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    boosted_tree.fit(soc_train, y_train)
    boosted_tree_preds = boosted_tree.predict_proba(soc_test)[:, 1]

    # Trenowanie sieci neuronowej
    neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    neural_net.fit(demo_train, y_train)
    neural_net_preds = neural_net.predict_proba(demo_test)[:, 1]

    # Połącz predykcje w DataFrame
    combined_preds = pd.DataFrame({
        'boosted_tree': boosted_tree_preds,
        'neural_net': neural_net_preds
    })

    # Trenowanie regresji logistycznej
    logistic_reg = LogisticRegression()
    logistic_reg.fit(combined_preds, y_test)

    return logistic_reg, combined_preds, y_test
