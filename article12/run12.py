from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np

def metoda12(X_train, y_train, X_test, y_test):
    """
    Trenuje i ocenia klasyfikatory Random Forest oraz CatBoost.

    Parametry:
        X_train (array-like): Cechy zbioru treningowego.
        y_train (array-like): Etykiety zbioru treningowego.
        X_test (array-like): Cechy zbioru testowego.
        y_test (array-like): Etykiety zbioru testowego.

    Zwraca:
        dict: Słownik zawierający wytrenowane modele oraz odpowiadające 
              im metryki oceny dla zbioru testowego.
    """
    # Trening Random Forest (Bagging)
    print("Trening Random Forest...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        max_features='sqrt', 
        random_state=42,
        criterion='gini',
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True
    ) # [cite: 283, 284, 285]
    rf_classifier.fit(X_train, y_train)
    rf_y_pred = rf_classifier.predict(X_test)

    # Trening CatBoost (Boosting)
    print("Trening CatBoost...")
    catboost_classifier = CatBoostClassifier(
        iterations=200,
        learning_rate=0.01,
        eval_metric='Accuracy',
        early_stopping_rounds=20,
        use_best_model=True,
        verbose=50
    ) # [cite: 285, 286, 287]
    catboost_classifier.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)
    catboost_y_pred = catboost_classifier.predict(X_test)

    def evaluate_model(y_true, y_pred, model_name):
        """Oblicza i wypisuje metryki oceny dla danego modelu."""

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        print(f"\n--- {model_name} Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }

    rf_metrics = evaluate_model(y_test, rf_y_pred, "Random Forest")
    catboost_metrics = evaluate_model(y_test, catboost_y_pred, "CatBoost")

    # Zwróć oba modele wraz z metrykami oceny
    return {
        "RandomForest": {"model": rf_classifier, "metrics": rf_metrics},
        "CatBoost": {"model": catboost_classifier, "metrics": catboost_metrics}
    }