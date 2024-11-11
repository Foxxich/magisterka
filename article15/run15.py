import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from common import load_and_preprocess_data, vectorize_data, split_data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_run15():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Tokenizacja i padding tekstu
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 300

    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, maxlen=maxlen)

    # Podział danych
    X_train, X_test, y_train, y_test = split_data(X_padded, y, test_size=0.2)

    # Przygotowanie osadzeń (losowe osadzenia dla celów demonstracyjnych)
    embedding_dim = 100
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)

    # Budowa modelu Bi-LSTM
    input_text = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)(input_text)
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    lstm_output = GlobalAveragePooling1D()(lstm_layer)

    dense_1 = Dense(128, activation='relu')(lstm_output)
    dropout = Dropout(0.5)(dense_1)
    output = Dense(1, activation='sigmoid')(dropout)

    # Kompilacja modelu
    model = Model(inputs=input_text, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Wczesne zatrzymanie
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Trenowanie modelu
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    return model, X_test, y_test
