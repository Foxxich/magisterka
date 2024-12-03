from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np

def create_model(input_dim, units1=128, units2=64, units3=32, dropout_rate=0.2, learning_rate=0.001, l2_reg=0.01):
    """
    Tworzy model sieci neuronowej z podanymi hiperparametrami.
    """
    model = Sequential([
        Dense(units1, input_dim=input_dim, activation=None, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout_rate),
        Dense(units2, activation=None, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout_rate),
        Dense(units3, activation=None, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_run1(X_train, y_train, X_test, y_test, n_models=5):
    """
    Trenuje ensemble modeli neuronowych z różnymi hiperparametrami.
    
    Parameters:
        X_train (np.ndarray): Dane treningowe.
        y_train (np.ndarray): Etykiety treningowe.
        X_test (np.ndarray): Dane testowe.
        y_test (np.ndarray): Etykiety testowe.
        n_models (int): Liczba modeli w ensemble.
    
    Returns:
        models: Lista wytrenowanych modeli.
        predictions: Uśrednione przewidywania na danych testowych.
    """
    models = []
    predictions = []

    # Hiperparametry do eksploracji
    units1_list = [128, 256, 64]
    units2_list = [64, 128, 32]
    learning_rates = [0.001, 0.005, 0.0005]
    dropouts = [0.2, 0.3, 0.1]

    for i in range(n_models):
        # Wybierz losowe hiperparametry dla każdego modelu
        units1 = np.random.choice(units1_list)
        units2 = np.random.choice(units2_list)
        dropout_rate = np.random.choice(dropouts)
        learning_rate = np.random.choice(learning_rates)

        print(f"Model {i + 1}: units1={units1}, units2={units2}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")

        # Stwórz i trenuj model
        model = create_model(
            input_dim=X_train.shape[1],
            units1=units1,
            units2=units2,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[lr_scheduler],
            verbose=1
        )

        models.append(model)
        predictions.append(model.predict(X_test))

    # Uśrednianie wyników
    predictions = np.mean(predictions, axis=0)
    return models, X_test, y_test
