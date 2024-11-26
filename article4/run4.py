from common import split_data
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.optimizers import Adadelta
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_run4(X_train, y_train, X_test, y_test, embedding_dim=100):
    """
    Trains a Bi-LSTM and CNN hybrid model using the provided embeddings.
    
    Args:
        X_train (np.ndarray): Training set features.
        y_train (list or np.ndarray): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (list or np.ndarray): Test set labels.
        embedding_dim (int): Embedding dimension size (default: 100).
        
    Returns:
        model: Trained model.
        X_test_reshaped: Reshaped test set features.
        y_test: Test set labels.
    """
    # Ensure embeddings are 2D
    if len(X_train.shape) == 1:
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Determine input shape dynamically
    input_sequence_length = X_train.shape[1]
    if input_sequence_length < 5:
        raise ValueError(f"Input sequence length ({input_sequence_length}) is too short for Conv1D kernel size (5).")

    # Input layers
    input_layer = Input(shape=(input_sequence_length, 1))

    # LSTM branch
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    flatten_lstm = Flatten()(bi_lstm)

    # CNN branch
    conv = Conv1D(128, kernel_size=min(5, input_sequence_length), activation='relu')(input_layer)
    maxpool = MaxPooling1D(pool_size=2)(conv)  # Ensure pooling size is compatible
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
