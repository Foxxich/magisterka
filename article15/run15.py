import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_run15(X_train, y_train, X_test, y_test, embedding_dim=200, maxlen=256, epochs=5, batch_size=32):
    """
    Trains a Bi-LSTM model with tokenized and padded input sequences.

    Parameters:
        X_train (list, np.ndarray, or pd.Series): Training text data.
        y_train (array-like): Training target labels.
        X_test (list, np.ndarray, or pd.Series): Test text data.
        y_test (array-like): Test target labels.
        embedding_dim (int): Dimension of the embeddings (default: 200).
        maxlen (int): Maximum length of input sequences (default: 256).
        epochs (int): Number of training epochs (default: 5).
        batch_size (int): Size of the training batches (default: 32).

    Returns:
        model (Model): Trained Keras model.
        np.ndarray: Test data inputs (padded sequences).
        array-like: Test data labels.
    """
    # Debug print for input data

    # Ensure input text data is in the correct format
    def ensure_text_format(data):
        """
        Ensures that the input data is a list of strings.
        If data contains non-string elements, they are converted to strings.
        """
        if not isinstance(data, list):
            print("Warning: Converting non-list input data to list.")
            data = list(data)
        data = [str(item) if not isinstance(item, str) else item for item in data]
        return data

    # Validate and format text data
    X_train = ensure_text_format(X_train)
    X_test = ensure_text_format(X_test)

    # Tokenization and padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1

    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen)
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)

    # Prepare embedding matrix (randomly initialized for simplicity)
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)

    # Build the Bi-LSTM model
    input_text = Input(shape=(maxlen,))
    embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=True  # Allow embeddings to be updated during training
    )(input_text)
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    lstm_output = GlobalAveragePooling1D()(lstm_layer)

    dense_1 = Dense(256, activation='relu')(lstm_output)
    dropout = Dropout(0.4)(dense_1)
    output = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=input_text, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # Train the model
    model.fit(
        X_train_padded,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, X_test_padded, y_test
