from common import split_data
from transformers import BertTokenizer, TFBertModel
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_run6(X_embeddings, X, y, batch_size=32):
    """
    Trains a Voting Classifier using AdaBoost and Logistic Regression with BERT embeddings.
    
    Args:
        X_embeddings (np.ndarray or sparse matrix): Precomputed embeddings or features.
        X (list): Original text data (if needed for padding/tokenization).
        y (list): Target labels.
        batch_size (int): Batch size for embedding generation (default: 32).
        
    Returns:
        model: Trained voting classifier.
        X_test_bert: Test set embeddings.
        y_test: Test set labels.
    """
    # Check if embeddings are already provided; if not, generate BERT embeddings
    if X_embeddings is None:
        # Initialize BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')

        def embed_text_in_batches(texts):
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(batch_texts.tolist(), return_tensors="tf", padding=True, truncation=True, max_length=128)
                outputs = bert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                embeddings.append(outputs.pooler_output.numpy())
            return np.vstack(embeddings)

        # Generate BERT embeddings
        X_embeddings = embed_text_in_batches(X)

    # Split embeddings into train and test sets
    X_train_bert, X_test_bert, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

    # Define base classifiers
    ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

    # Define ensemble classifier with soft voting
    voting_classifier = VotingClassifier(
        estimators=[('ada', ada_boost), ('lr', logistic_regression)],
        voting='soft'
    )

    # Train the ensemble model
    voting_classifier.fit(X_train_bert, y_train)

    return voting_classifier, X_test_bert, y_test
