from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Embedding, Reshape, ZeroPadding1D
from keras.optimizers import Adadelta
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.utils import pad_sequences # Import pad_sequences

def metoda4(X_train, y_train, X_test, y_test, embedding_dim=100):
    """
    Trenuje architekturę opartą na zespole sieci Bi-LSTM, CNN i MLP.
    Zachowuje oryginalną sygnaturę metody, ale modyfikuje wnętrze.

    Argumenty:
        X_train (np.ndarray): Zbiór cech do trenowania.
        y_train (list lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Zbiór cech testowych.
        y_test (list lub np.ndarray): Etykiety zbioru testowego.
        embedding_dim (int): Rozmiar wymiaru osadzeń (domyślnie: 100).

    Zwraca:
        model: Wytrenowany model.
        X_test_reshaped: Zmodyfikowany zbiór cech testowych.
        y_test: Etykiety zbioru testowego.
    """

    print("Początek metody metoda4")
    print(f"Kształt X_train przed przetworzeniem: {X_train.shape}")
    print(f"Kształt X_test przed przetworzeniem: {X_test.shape}")

    # Upewnij się, że dane wejściowe są dwuwymiarowe
    if len(X_train.shape) == 1:
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        print("Dane wejściowe zostały rozszerzone do 2D.")
        print(f"Kształt X_train po rozszerzeniu: {X_train.shape}")
        print(f"Kształt X_test po rozszerzeniu: {X_test.shape}")

    # Skalowanie cech
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Przeskalowane dane treningowe
    X_test_scaled = scaler.transform(X_test)  # Przeskalowane dane testowe
    print("Cechy zostały przeskalowane.")
    print(f"Kształt X_train_scaled: {X_train_scaled.shape}")
    print(f"Kształt X_test_scaled: {X_test_scaled.shape}")

    # Dynamiczne określenie długości sekwencji wejściowej
    input_sequence_length = X_train.shape[1]
    print(f"Długość sekwencji wejściowej: {input_sequence_length}")
    if input_sequence_length < 5:
        raise ValueError(f"Długość sekwencji wejściowej ({input_sequence_length}) jest za krótka dla jądra Conv1D (5).")

    # Warstwa wejściowa
    input_layer = Input(shape=(input_sequence_length, 1))
    print(f"Kształt warstwy wejściowej: {input_layer.shape}")

    # --- ŚCIEŻKA 1: Bi-LSTM i CNN na danych wejściowych ---
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    print(f"Kształt wyjścia Bi-LSTM: {bi_lstm.shape}")
    cnn = Conv1D(128, kernel_size=min(5, input_sequence_length), activation='relu')(input_layer)
    print(f"Kształt wyjścia CNN: {cnn.shape}")
    maxpool = MaxPooling1D(pool_size=2)(cnn)
    print(f"Kształt wyjścia MaxPooling: {maxpool.shape}")
    flatten_cnn = Flatten()(maxpool) # Flatten CNN output
    print(f"Kształt wyjścia Flatten CNN: {flatten_cnn.shape}")

    # Calculate padding size
    padding_size = input_sequence_length - flatten_cnn.shape[1]

    # Pad the CNN output to match Bi-LSTM sequence length
    if padding_size > 0:
      padded_cnn = np.pad(flatten_cnn, ((0, 0), (0, padding_size)), 'constant')
    else:
      padded_cnn = flatten_cnn

    flatten_lstm = Flatten()(bi_lstm) # Flatten Bi-LSTM output
    print(f"Kształt wyjścia Flatten BiLSTM: {flatten_lstm.shape}")
    flatten = concatenate([flatten_lstm, padded_cnn])  # Połączenie wyjść Bi-LSTM i CNN
    print(f"Kształt wyjścia Flatten (połączenie Bi-LSTM i CNN): {flatten.shape}")

    # --- ŚCIEŻKA 2: Prosta Dense na oryginalnych cechach ---
    dense_original = Dense(64, activation='relu')(input_layer) # Używamy input_layer, bo to oryginalne cechy
    print(f"Kształt wyjścia Dense (oryginalne cechy): {dense_original.shape}")
    flatten_dense_original = Flatten()(dense_original)
    print(f"Kształt wyjścia Flatten (oryginalne cechy): {flatten_dense_original.shape}")

    # --- Połączenie ścieżek ---
    #  Dodajemy Reshape, aby ujednolicić wymiary przed połączeniem
    reshape_flatten = Reshape((flatten.shape[1], 1))(flatten)
    print(f"Kształt po Reshape (flatten): {reshape_flatten.shape}")
    reshape_flatten_dense_original = Reshape((flatten_dense_original.shape[1], 1))(flatten_dense_original)
    print(f"Kształt po Reshape (flatten_dense_original): {reshape_flatten_dense_original.shape}")

    merged = concatenate([reshape_flatten, reshape_flatten_dense_original], axis=1) # Połączono wzdłuż osi 1 (cechy)
    print(f"Kształt po Concatenate: {merged.shape}")
    merged_flat = Flatten()(merged) # Spłaszczamy po połączeniu
    print(f"Kształt po Flatten (po Concatenate): {merged_flat.shape}")
    mlp_layer = Dense(128, activation='relu')(merged_flat)
    print(f"Kształt wyjścia MLP: {mlp_layer.shape}")
    output = Dense(1, activation='softmax')(mlp_layer)
    print(f"Kształt warstwy wyjściowej: {output.shape}")

    # Definicja modelu
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    # Dostosowanie kształtu danych do kompatybilności z modelem
    X_train_reshaped = np.expand_dims(X_train_scaled, axis=2) # Używamy przeskalowanych danych
    X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)  # Używamy przeskalowanych danych
    print("Dane wejściowe zostały zmodyfikowane.")
    print(f"Kształt X_train_reshaped: {X_train_reshaped.shape}")
    print(f"Kształt X_test_reshaped: {X_test_reshaped.shape}")

    # Trenowanie modelu
    model.fit(X_train_reshaped, y_train, epochs=3, batch_size=256, validation_data=(X_test_reshaped, y_test))

    print("Koniec metody metoda4")
    return model, X_test_reshaped, y_test