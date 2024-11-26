from transformers import BertTokenizer, TFBertModel
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_run6(X_train, y_train, X_test, y_test, batch_size=32, use_bert_embeddings=False):
    """
    Trains a Voting Classifier using AdaBoost and Logistic Regression with BERT embeddings.
    
    Args:
        X_train (np.ndarray or list): Training set features or original text data.
        y_train (list): Training set labels.
        X_test (np.ndarray or list): Test set features or original text data.
        y_test (list): Test set labels.
        batch_size (int): Batch size for embedding generation (default: 32).
        use_bert_embeddings (bool): Whether to generate BERT embeddings from raw text data.
        
    Returns:
        model: Trained voting classifier.
        X_test: Test set embeddings (if generated).
        y_test: Test set labels.
    """
    # Generate BERT embeddings if needed
    if use_bert_embeddings:
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

        # Generate embeddings for training and test sets
        X_train = embed_text_in_batches(X_train)
        X_test = embed_text_in_batches(X_test)

    # Define base classifiers
    ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

    # Define ensemble classifier with soft voting
    voting_classifier = VotingClassifier(
        estimators=[('ada', ada_boost), ('lr', logistic_regression)],
        voting='soft'
    )

    # Train the ensemble model
    voting_classifier.fit(X_train, y_train)

    return voting_classifier, X_test, y_test
