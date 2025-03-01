import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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
    Trenuje model zespołowy stacking z użyciem Random Forest, AdaBoost i Logistic Regression
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
    # Upewnij się, że dane wejściowe są 2-wymiarowe
    if len(X_train.shape) != 2 or len(X_test.shape) != 2:
        raise ValueError("Cechy wejściowe muszą być 2-wymiarowymi tablicami.")

    # Definicja modeli bazowych i meta-modelu do stacking
    print("Definiowanie modeli...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(max_iter=200, random_state=42)

    # Tworzenie klasyfikatora stacking
    print("Budowanie modelu stacking...")
    stack_model = StackingClassifier(
        estimators=[('rf', rf_model), ('ab', ab_model)],
        final_estimator=lr_model
    )

    # Trening modelu stacking
    print("Trening modelu stacking...")
    stack_model.fit(X_train, y_train)

    print("Trening zakończony.")

    # Zwróć wytrenowany model oraz dane testowe
    return stack_model, X_test, y_test
