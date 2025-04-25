import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk.stem import SnowballStemmer


def preprocess_text(text):
    """
    Oczyszcza i przetwarza tekst poprzez usunięcie URL-i, znaków specjalnych, pojedynczych liter i liczb,
    oraz stosuje stemming przy użyciu Snowball Stemmer.
    """
    snowball_stemmer = SnowballStemmer('english')
    text = re.sub(r'http\S+', '', text)  # Usuwanie URL-i
    text = re.sub(r'\W', ' ', text)  # Usuwanie znaków specjalnych
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Usuwanie pojedynczych liter
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Usuwanie pojedynczych liter na początku
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Zastąpienie wielu spacji jedną
    text = re.sub(r'^\s+|\s+?$', '', text.lower())  # Usuwanie spacji na początku/końcu i konwersja do małych liter
    return ' '.join([snowball_stemmer.stem(word) for word in text.split()])  # Zastosowanie stemmingu


def metoda14(X_train, y_train, X_test, y_test):
    """
    Trenuje model zespołowy stacking z użyciem Random Forest, AdaBoost, SVC i Logistic Regression
    jako finalnego estymatora.

    Parametry:
        X_train (numpy.ndarray): Cechy zbioru treningowego.
        y_train (numpy.ndarray): Etykiety zbioru treningowego.
        X_test (numpy.ndarray): Cechy zbioru testowego.
        y_test (numpy.ndarray): Etykiety zbioru testowego.

    Zwraca:
        StackingClassifier: Wytrenowany model stacking.
        numpy.ndarray: Cechy zbioru testowego.
        numpy.ndarray: Etykiety zbioru testowego.
    """
    print("Definiowanie modeli bazowych...")
    
    # Inicjalizacja klasyfikatorów bazowych
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
    svc_model = SVC(kernel='linear', probability=True, random_state=42)
    
    base_learners = [
        ('rf', rf_model),
        ('ab', ab_model),
        ('svc', svc_model)
    ]
    
    # Inicjalizacja meta-klasyfikatora
    meta_classifier = LogisticRegression(random_state=42)
    
    print("Budowanie i trenowanie modelu Stacking...")
    
    # Inicjalizacja i trening modelu Stacking
    stack_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_classifier,
        cv=5  # Użycie cross-walidacji (5-krotnej)
    )
    
    stack_model.fit(X_train, y_train)
    
    print("Trening zakończony.")
    
    return stack_model, X_test, y_test