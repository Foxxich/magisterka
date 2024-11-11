from common import load_and_preprocess_data, vectorize_data, split_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, accuracy_score

def train_run1():
    # Przygotowanie danych
    X, y = load_and_preprocess_data()
    X_tfidf, _ = vectorize_data(X, max_features=5000)
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.2)

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
        callbacks=[lr_scheduler]
    )

    return model, X_test, y_test
