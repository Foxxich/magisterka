from transformers import BertTokenizer, TFBertModel
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def metoda6(X_train, y_train, X_test, y_test, batch_size=32, use_bert_embeddings=False):
    """
    Trenuje klasyfikator zespołowy Voting Classifier z wykorzystaniem AdaBoost i Logistic Regression oraz osadzeń BERT.
    
    Parametry:
        X_train (np.ndarray lub lista): Cechy zbioru treningowego lub oryginalne dane tekstowe.
        y_train (lista): Etykiety zbioru treningowego.
        X_test (np.ndarray lub lista): Cechy zbioru testowego lub oryginalne dane tekstowe.
        y_test (lista): Etykiety zbioru testowego.
        batch_size (int): Rozmiar batcha do generowania osadzeń (domyślnie: 32).
        use_bert_embeddings (bool): Czy generować osadzenia BERT z surowych danych tekstowych.
        
    Zwraca:
        model: Wytrenowany klasyfikator zespołowy.
        X_test: Osadzenia zbioru testowego (jeśli wygenerowane).
        y_test: Etykiety zbioru testowego.
    """
    # Generuj osadzenia BERT, jeśli wymagane
    if use_bert_embeddings:
        # Inicjalizuj tokenizer i model BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')

        def embed_text_in_batches(texts):
            # Generuj osadzenia w batchach
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(batch_texts.tolist(), return_tensors="tf", padding=True, truncation=True, max_length=128)
                outputs = bert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                embeddings.append(outputs.pooler_output.numpy())
            return np.vstack(embeddings)

        # Generuj osadzenia dla zbiorów treningowego i testowego
        X_train = embed_text_in_batches(X_train)
        X_test = embed_text_in_batches(X_test)

    # Definiuj klasyfikatory bazowe
    ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

    # Definiuj klasyfikator zespołowy z miękkim głosowaniem
    voting_classifier = VotingClassifier(
        estimators=[('ada', ada_boost), ('lr', logistic_regression)],
        voting='soft'
    )

    # Trenuj model zespołowy
    voting_classifier.fit(X_train, y_train)

    return voting_classifier, X_test, y_test
