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
    Funkcja trenuje zespół modeli Bi-LSTM z tokenizowanymi i wyściełanymi sekwencjami wejściowymi.

    Parametry:
        X_train (list, np.ndarray lub pd.Series): Dane tekstowe do trenowania.
        y_train (array-like): Etykiety docelowe dla danych treningowych.
        X_test (list, np.ndarray lub pd.Series): Dane tekstowe do testowania.
        y_test (array-like): Etykiety docelowe dla danych testowych.
        embedding_dim (int): Wymiar osadzania (domyślnie: 200).
        maxlen (int): Maksymalna długość sekwencji wejściowych (domyślnie: 256).
        epochs (int): Liczba epok treningowych (domyślnie: 5).
        batch_size (int): Rozmiar partii danych podczas trenowania (domyślnie: 32).

    Zwraca:
        list: Lista wytrenowanych modeli Keras.
        np.ndarray: Dane testowe jako wyściełane sekwencje.
        array-like: Etykiety dla danych testowych.
    """
    # Funkcja pomocnicza: zapewnia, że dane wejściowe są w formacie tekstowym
    def ensure_text_format(data):
        if not isinstance(data, list):
            print("Ostrzeżenie: Konwersja danych wejściowych do listy.")
            data = list(data)
        data = [str(item) if not isinstance(item, str) else item for item in data]
        return data

    # Walidacja i formatowanie danych tekstowych
    X_train = ensure_text_format(X_train)
    X_test = ensure_text_format(X_test)

    # Tokenizacja i wyściełanie sekwencji
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1  # Liczba unikalnych słów w słowniku +1

    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen)
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)

    # Przygotowanie macierzy osadzeń (inicjalizowane losowo dla uproszczenia)
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)

    # Funkcja budująca model Bi-LSTM
    def build_model():
        input_text = Input(shape=(maxlen,))
        embedding_layer = Embedding(
            vocab_size,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=True
        )(input_text)
        lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
        lstm_output = GlobalAveragePooling1D()(lstm_layer)

        dense_1 = Dense(256, activation='relu')(lstm_output)
        dropout = Dropout(0.4)(dense_1)
        output = Dense(1, activation='sigmoid')(dropout)

        model = Model(inputs=input_text, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Trenowanie zespołu modeli
    models = []
    for i in range(3):  # Tworzenie zespołu z 3 modeli
        model = build_model()
        print(f"Trenowanie modelu {i + 1}...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        model.fit(
            X_train_padded,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        models.append(model)

    return models, X_test_padded, y_test
