from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def metoda2(X_train, y_train, X_test, y_test):
    """
    Trenuje model wykorzystując GradientBoosting oraz MLP na oddzielnych podzbiorach cech 
    i łączy je za pomocą Regresji Logistycznej.
    
    Parametry:
        X_train (np.ndarray lub DataFrame): Zbiór cech do trenowania.
        y_train (list lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray lub DataFrame): Zbiór cech do testowania.
        y_test (list lub np.ndarray): Etykiety zbioru testowego.
    
    Zwraca:
        logistic_reg: Wytrenowany model regresji logistycznej.
        combined_preds_test: Połączone predykcje (cechy) użyte dla meta-modelu (zbiór testowy).
        y_test: Etykiety zbioru testowego.
    """
    # Upewnij się, że dane wejściowe cech są w formacie DataFrame
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    # Podział cechy na podzbiory: demograficzne i zachowania społeczne
    num_features = X_train_df.shape[1]
    if num_features < 6:
        raise ValueError("Zbyt mała liczba cech, aby podzielić na podzbiory demograficzne i zachowania społeczne.")
    demographic_features_train = X_train_df.iloc[:, :5]
    social_behavior_features_train = X_train_df.iloc[:, 5:]
    demographic_features_test = X_test_df.iloc[:, :5]
    social_behavior_features_test = X_test_df.iloc[:, 5:]

    # Trenuj GradientBoostingClassifier na cechach zachowań społecznych
    boosted_tree = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    boosted_tree.fit(social_behavior_features_train, y_train)
    boosted_tree_preds_test = boosted_tree.predict_proba(social_behavior_features_test)[:, 1]

    # Trenuj MLPClassifier na cechach demograficznych
    neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    neural_net.fit(demographic_features_train, y_train)
    neural_net_preds_test = neural_net.predict_proba(demographic_features_test)[:, 1]

    # Połącz predykcje w jeden DataFrame
    combined_preds_test = pd.DataFrame({
        'boosted_tree': boosted_tree_preds_test,
        'neural_net': neural_net_preds_test
    })

    # Trenuj Regresję Logistyczną na połączonych predykcjach
    logistic_reg = LogisticRegression()
    logistic_reg.fit(combined_preds_test, y_test)

    return logistic_reg, combined_preds_test, y_test
