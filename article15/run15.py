import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_run15(X_embeddings=None, X=None, y=None, embedding_dim=100, maxlen=300, epochs=10, batch_size=64):
    """
    Trains a Bi-LSTM model with tokenized and padded input sequences.

    Parameters:
        X_embeddings (np.ndarray): Precomputed embeddings (not used for this implementation).
        X (list or Series): Input text data.
        y (array-like): Target labels.
        embedding_dim (int): Dimension of the embeddings.
        maxlen (int): Maximum length of input sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Size of the training batches.
    
    Returns:
        model (Model): Trained Keras model.
        X_test (np.ndarray): Test data inputs.
        y_test (array-like): Test data labels.
    """
    # Validate input
    if X is None or y is None:
        raise ValueError("X and y must be provided for train_run15.")
    
    # Tokenization and padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, maxlen=maxlen)

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # Prepare embedding matrix (randomly initialized for simplicity)
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)

    # Build the Bi-LSTM model
    input_text = Input(shape=(maxlen,))
    embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=True  # Set trainable to True for updating embeddings during training
    )(input_text)
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    lstm_output = GlobalAveragePooling1D()(lstm_layer)

    dense_1 = Dense(128, activation='relu')(lstm_output)
    dropout = Dropout(0.5)(dense_1)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=input_text, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    return model, X_test, y_test
