from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Softmax
from keras.optimizers import Adadelta
from sklearn.preprocessing import StandardScaler
import numpy as np

def metoda4(X_train, y_train, X_test, y_test, embedding_dim=100):
    """
    Trenuje architekturę opartą na zespole sieci Bi-LSTM, CNN i MLP z klasyfikatorem Softmax.
    
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
    # Upewnij się, że dane wejściowe są dwuwymiarowe
    if len(X_train.shape) == 1:
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)

    # Skalowanie cech
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Dynamiczne określenie długości sekwencji wejściowej
    input_sequence_length = X_train.shape[1]
    if input_sequence_length < 5:
        raise ValueError(f"Długość sekwencji wejściowej ({input_sequence_length}) jest za krótka dla jądra Conv1D (5).")

    # Warstwa wejściowa
    input_layer = Input(shape=(input_sequence_length, 1))

    # Sieć 1: Bi-LSTM
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    flatten_lstm = Flatten()(bi_lstm)

    # Sieć 2: CNN
    conv = Conv1D(128, kernel_size=min(5, input_sequence_length), activation='relu')(input_layer)
    maxpool = MaxPooling1D(pool_size=2)(conv)
    flatten_cnn = Flatten()(maxpool)

    # Sieć n: Dense (sieć MLP jako demonstracja)
    dense_net = Dense(128, activation='relu')(input_layer)
    flatten_dense = Flatten()(dense_net)

    # Połączenie wyjść z wszystkich sieci
    merged = concatenate([flatten_lstm, flatten_cnn, flatten_dense])
    mlp_layer = Dense(128, activation='relu')(merged)

    # Warstwa wyjściowa z aktywacją sigmoidalną
    output = Dense(1, activation='sigmoid')(mlp_layer)

    # Definicja modelu
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    # Dostosowanie kształtu danych do kompatybilności z modelem
    X_train_reshaped = np.expand_dims(X_train, axis=2)
    X_test_reshaped = np.expand_dims(X_test, axis=2)

    # Trenowanie modelu
    model.fit(X_train_reshaped, y_train, epochs=3, batch_size=256, validation_data=(X_test_reshaped, y_test))

    return model, X_test_reshaped, y_test
