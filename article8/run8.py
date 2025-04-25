from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier


def create_dbn_model(input_dim, num_classes):
    """
    Tworzy model DBN jako prostą głęboką sieć neuronową z Keras.
    """
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def metoda8(X_train, y_train, X_test, y_test):
    """
    Trenuje klasyfikator zespołowy Voting Classifier z użyciem RandomForest, XGBoost i DBN.

    Parametry:
        X_train (np.ndarray): Cechy zbioru treningowego.
        y_train (lista lub np.ndarray): Etykiety zbioru treningowego.
        X_test (np.ndarray): Cechy zbioru testowego.
        y_test (lista lub np.ndarray): Etykiety zbioru testowego.

    Zwraca:
        ensemble_model: Wytrenowany klasyfikator zespołowy.
        X_test: Cechy zbioru testowego.
        y_test: Etykiety zbioru testowego.
    """
    # Konwertuj macierz rzadką na gęstą, jeśli to konieczne
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Skalowanie danych dla DBN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Określ liczbę cech i klas
    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))

    # Definicja klasyfikatorów bazowych
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    dbn = KerasClassifier(model=lambda: create_dbn_model(input_dim, num_classes), epochs=50, batch_size=32, verbose=0)

    # Definicja modelu zespołowego
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('dbn', dbn)],
        voting='soft'
    )

    # Trening modelu zespołowego
    ensemble_model.fit(X_train_scaled, y_train)

    return ensemble_model, X_test, y_test