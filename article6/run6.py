from common import load_and_preprocess_data, evaluate_model, vectorize_data, split_data
from transformers import BertTokenizer, TFBertModel
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

def train_run6():
    # Krok 1: Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Krok 2: Podziel zbiór danych na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Krok 3: Zainicjalizuj tokenizer i model BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def embed_text_in_batches(texts, batch_size=32):
        # Wyznaczanie osadzeń (embeddings) w partiach
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts.tolist(), return_tensors="tf", padding=True, truncation=True, max_length=128)
            outputs = bert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            embeddings.append(outputs.pooler_output.numpy())
        return np.vstack(embeddings)

    # Generowanie osadzeń BERT dla danych treningowych i testowych w partiach
    X_train_bert = embed_text_in_batches(X_train, batch_size=32)
    X_test_bert = embed_text_in_batches(X_test, batch_size=32)

    # Krok 4: Definiowanie klasyfikatorów bazowych
    ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

    # Krok 5: Definiowanie klasyfikatora zespołowego z głosowaniem miękkim
    voting_classifier = VotingClassifier(
        estimators=[('ada', ada_boost), ('lr', logistic_regression)],
        voting='soft'
    )

    # Trenowanie modelu zespołowego
    voting_classifier.fit(X_train_bert, y_train)

    # Zwróć wytrenowany model i dane testowe
    return voting_classifier, X_test_bert, y_test
