from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier
import numpy as np

def metoda13(X_train, y_train, X_test, y_test):
    """
    Trenuje model zespołowy (stacking) do obsługi problemu niezrównoważonych klas.

    Parametry:
        X_train (np.ndarray): Macierz cech dla danych treningowych.
        y_train (np.ndarray): Wektor etykiet dla danych treningowych.
        X_test (np.ndarray): Macierz cech dla danych testowych.
        y_test (np.ndarray): Wektor etykiet dla danych testowych.

    Zwraca:
        StackingClassifier: Wytrenowany model zespołowy typu stacking.
        np.ndarray: Dane testowe (macierz cech).
        np.ndarray: Dane testowe (etykiety).
    """
    # Obsługa niezrównoważonych klas za pomocą SMOTETomek
    # Kombinacja nadpróbkowania SMOTE
    smotetomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smotetomek.fit_resample(X_train, y_train)
    
    # Definiowanie bazowych klasyfikatorów (uczestnicy zespołu)
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),  # Random Forest
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),  # Gradient Boosting
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, reg_alpha=1.0, reg_lambda=1.0))
    ]
    
    # Definiowanie klasyfikatora meta (uczeń końcowy w stacking)
    meta_learner = LogisticRegression(random_state=42)
    
    # Tworzenie modelu stacking (łączenie bazowych klasyfikatorów)
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
    
    # Trenowanie modelu stacking na zbalansowanych danych
    stacking_model.fit(X_train_resampled, y_train_resampled)
    
    # Ocena modelu na danych testowych
    y_pred = stacking_model.predict(X_test)
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))
    
    return stacking_model, X_test, y_test