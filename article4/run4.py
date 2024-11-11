from common import load_and_preprocess_data, split_data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.optimizers import Adadelta

def train_run4():
    # Wczytaj i przetwórz dane
    X, y = load_and_preprocess_data()

    # Tokenizacja i padding
    MAX_NB_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 300
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    word_index = tokenizer.word_index

    X_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Podział na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = split_data(X_padded, y, test_size=0.2)

    # Definicja modelu Bi-LSTM i CNN
    input_lstm = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_lstm = Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_lstm)
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_lstm)
    flatten_lstm = Flatten()(bi_lstm)

    input_cnn = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_cnn = Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_cnn)
    conv = Conv1D(128, 5, activation='relu')(embedding_cnn)
    maxpool = MaxPooling1D(pool_size=5)(conv)
    flatten_cnn = Flatten()(maxpool)

    merged = concatenate([flatten_lstm, flatten_cnn])
    dense_1 = Dense(128, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(dense_1)

    model = Model(inputs=[input_lstm, input_cnn], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    # Trenowanie modelu
    model.fit([X_train, X_train], y_train, epochs=3, batch_size=256, validation_data=([X_test, X_test], y_test))

    return model, [X_test, X_test], y_test
