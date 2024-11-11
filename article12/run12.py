from common import load_and_preprocess_data, vectorize_data, split_data, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

def train_run12():
    # Wczytaj i przetwórz zbiór danych
    X, y = load_and_preprocess_data()

    # Wektoryzuj dane za pomocą TF-IDF
    X_vectorized, tfidf = vectorize_data(X, max_features=10000)

    # Podziel dane na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_vectorized, y, test_size=0.2)

    # Random Forest (Bagging)
    rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    rf_classifier.fit(X_train, y_train)

    # CatBoost (Boosting)
    catboost_classifier = CatBoostClassifier(
        iterations=200, learning_rate=0.01, eval_metric='Accuracy',
        early_stopping_rounds=20, use_best_model=True, verbose=50
    )
    catboost_classifier.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

    # Zwróć oba modele do oceny
    return {
        "RandomForest": (rf_classifier, X_test, y_test),
        "CatBoost": (catboost_classifier, X_test, y_test)
    }
