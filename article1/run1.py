from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np

def train_run1(X_train, y_train, X_test, y_test):
    """
    Trains a neural network using the provided training and testing sets.
    
    Parameters:
        X_train (np.ndarray): Training set features.
        y_train (np.ndarray): Training set labels.
        X_test (np.ndarray): Test set features.
        y_test (np.ndarray): Test set labels.
    
    Returns:
        model: Trained Keras model.
        X_test: Test set features.
        y_test: Test set labels.
    """
    # Definicja modelu
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation=None, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        Dense(64, activation=None, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        Dense(32, activation=None, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Scheduler dla zmiany tempa uczenia
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    # Trenowanie modelu
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[lr_scheduler],
        verbose=1
    )

    return model, X_test, y_test
