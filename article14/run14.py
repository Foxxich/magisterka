import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import SnowballStemmer
from common import load_and_preprocess_data, vectorize_data, split_data


def preprocess_text(text):
    # Usuń adresy URL, znaki specjalne, pojedyncze litery, liczby oraz zastosuj stemming
    snowball_stemmer = SnowballStemmer('english')
    text = re.sub(r'http\S+', '', text)  # Usuń URL
    text = re.sub(r'\W', ' ', text)  # Usuń znaki specjalne
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Usuń pojedyncze litery
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Usuń znaki na początku
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Zastąp wielokrotne spacje pojedynczymi
    text = re.sub(r'^\s+|\s+?$', '', text.lower())  # Usuń spacje na początku i końcu oraz zamień na małe litery
    return ' '.join([snowball_stemmer.stem(word) for word in text.split()])  # Zastosuj stemming


def train_run14():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Zastosuj dodatkowe przetwarzanie tekstu
    X = X.apply(preprocess_text)

    # Wektoryzacja tekstu
    X_tfidf, tfidf_vectorizer = vectorize_data(X, max_features=5000)

    # Podział danych na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.2)

    # Zdefiniuj modele bazowe oraz meta-model dla stacking
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(max_iter=200, random_state=42)

    # Stwórz klasyfikator stacking
    stack_model = StackingClassifier(
        estimators=[('rf', rf_model), ('ab', ab_model)],
        final_estimator=lr_model
    )

    # Wytrenuj model stacking
    stack_model.fit(X_train, y_train)

    # Zwróć model oraz dane testowe
    return stack_model, X_test, y_test
