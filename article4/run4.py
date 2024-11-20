from common import split_data
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.optimizers import Adadelta
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_run4(X_embeddings, X, y, max_sequence_length=300, embedding_dim=100):
    """
    Trains a Bi-LSTM and CNN hybrid model using the provided embeddings.
    
    Args:
        X_embeddings (np.ndarray or sparse matrix): Precomputed embeddings or features.
        X (list): Original text data (if needed for padding/tokenization).
        y (list): Target labels.
        max_sequence_length (int): Maximum sequence length for embeddings (default: 300).
        embedding_dim (int): Embedding dimension size (default: 100).
        
    Returns:
        model: Trained model.
        X_test: Test set features.
        y_test: Test set labels.
    """
    # Check if the embeddings are dense or sparse
    if hasattr(X_embeddings, "toarray"):
        X_embeddings = X_embeddings.toarray()  # Convert sparse matrix to dense if necessary

    # Ensure the embeddings are 2D
    if len(X_embeddings.shape) == 1:
        X_embeddings = np.expand_dims(X_embeddings, axis=1)

    # Scale features if they are numeric
    scaler = StandardScaler()
    X_embeddings = scaler.fit_transform(X_embeddings)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X_embeddings, y, test_size=0.2)

    # Input layers
    input_layer = Input(shape=(X_train.shape[1], 1))  # Adjust for input shape

    # LSTM branch
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    flatten_lstm = Flatten()(bi_lstm)

    # CNN branch
    conv = Conv1D(128, 5, activation='relu')(input_layer)
    maxpool = MaxPooling1D(pool_size=5)(conv)
    flatten_cnn = Flatten()(maxpool)

    # Merge branches
    merged = concatenate([flatten_lstm, flatten_cnn])
    dense_1 = Dense(128, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    # Reshape data for model compatibility
    X_train_reshaped = np.expand_dims(X_train, axis=2)
    X_test_reshaped = np.expand_dims(X_test, axis=2)

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=3, batch_size=256, validation_data=(X_test_reshaped, y_test))

    return model, X_test_reshaped, y_test
