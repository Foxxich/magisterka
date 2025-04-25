from transformers import BertTokenizer, TFBertModel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input, Reshape  # Added Reshape
from scikeras.wrappers import KerasClassifier
import tensorflow as tf


def create_cnn_model(input_shape, num_classes):
    """
    Tworzy model CNN zgodny z konfiguracją z artykułu (Tabela 3 dla MVSA).
    """
    print(
        "Entering create_cnn_model with input_shape:",
        input_shape,
        "and num_classes:",
        num_classes,
    )
    model = Sequential()
    model.add(Input(shape=input_shape))  # Define input shape here
    print("Added Input layer, shape:", input_shape)

    # Add Reshape layer to explicitly make input 3D
    if len(input_shape) == 1:
        model.add(Reshape((input_shape[0], 1)))  # Reshape to (input_dim, 1)
        print("Added Reshape layer")

    model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
    print("Added Conv1D layer")
    model.add(MaxPooling1D(pool_size=5))
    print("Added MaxPooling1D layer")
    model.add(Conv1D(filters=128, kernel_size=5, activation="relu"))
    print("Added Conv1D layer")
    model.add(MaxPooling1D(pool_size=5))
    print("Added MaxPooling1D layer")
    model.add(Dropout(0.2))
    print("Added Dropout layer")
    model.add(Flatten())
    print("Added Flatten layer")
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
        print("Added Dense (sigmoid) layer")
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=["accuracy"],
        )
        print("Compiled model for binary classification")
    else:
        model.add(Dense(num_classes, activation="softmax"))
        print("Added Dense (softmax) layer")
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=["accuracy"],
        )
        print("Compiled model for multi-class classification")
    print("Exiting create_cnn_model")
    return model


def metoda6(X_train, y_train, X_test, y_test, batch_size=32, use_bert_embeddings=False):
    """
    Trenuje klasyfikator zespołowy WCNNE z AdaBoost na osadzeniach BERT, zgodnie z artykułem.

    Parametry:
        X_train (np.ndarray lub lista): Cechy zbioru treningowego lub oryginalne dane tekstowe.
        y_train (lista): Etykiety zbioru treningowego.
        X_test (np.ndarray lub lista): Cechy zbioru testowego lub oryginalne dane tekstowe.
        y_test (lista): Etykiety zbioru testowego.
        batch_size (int): Rozmiar batcha do generowania osadzeń (domyślnie: 32).
        use_bert_embeddings (bool): Czy generować osadzenia BERT z surowych danych tekstowych.

    Zwraca:
        model: Wytrenowany klasyfikator zespołowy.
        X_test: Osadzenia zbioru testowego (jeśli wygenerowane).
        y_test: Etykiety zbioru testowego.
    """
    print("Entering metoda6")

    # Generuj osadzenia BERT, jeśli wymagane
    if use_bert_embeddings:
        print("Using BERT embeddings")
        # Inicjalizuj tokenizer i model BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("Initialized tokenizer")
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        print("Initialized BERT model")

        def embed_text_in_batches(texts):
            print("Entering embed_text_in_batches")
            embeddings = []
            for i in range(0, len(texts), batch_size):
                print("Processing batch", i // batch_size)
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="tf",
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_attention_mask=True,
                )
                print("Tokenized batch, input_ids shape:", inputs["input_ids"].shape)
                outputs = bert_model(
                    inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                print(
                    "Got BERT outputs, last_hidden_state shape:",
                    outputs.last_hidden_state.shape,
                )
                # Użyj last_hidden_state zamiast pooler_output, aby zachować sekwencję
                embeddings.append(outputs.last_hidden_state.numpy())
                print("Appended embeddings")
            print("Concatenating embeddings")
            final_embeddings = np.concatenate(embeddings, axis=0)
            print(
                "Concatenated embeddings, final shape:",
                final_embeddings.shape,
            )  # Shape check
            print("Exiting embed_text_in_batches")
            return final_embeddings

        # Generuj osadzenia dla zbiorów treningowego i testowego
        print("Generating training embeddings")
        X_train = embed_text_in_batches(X_train)
        print("Generated training embeddings, shape:", X_train.shape)
        print("Generating testing embeddings")
        X_test = embed_text_in_batches(X_test)
        print("Generated testing embeddings, shape:", X_test.shape)

    # Skalowanie danych
    print("Scaling data")
    scaler = StandardScaler()
    # Reshape only if X_train is 3D, otherwise, leave it as is
    if X_train.ndim == 3:
        print("X_train is 3D, reshaping for scaling")
        X_train_scaled = scaler.fit_transform(
            X_train.reshape(X_train.shape[0], -1)
        ).reshape(X_train.shape)
        X_test_scaled = scaler.transform(
            X_test.reshape(X_test.shape[0], -1)
        ).reshape(X_test.shape)
        input_shape = (
            X_train_scaled.shape[1],
            X_train_scaled.shape[2],
        )  # (max_length, embedding_dim)
        print("Reshaped and scaled data, new shape:", X_train_scaled.shape)
    else:
        print("X_train is 2D, scaling directly")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        input_shape = (X_train_scaled.shape[1],)  # (embedding_dim,)
        print("Scaled data, new shape:", X_train_scaled.shape)

    # Określ liczbę klas
    num_classes = len(np.unique(y_train))
    print("Number of classes:", num_classes)

    print("X_train_scaled shape:", X_train_scaled.shape)  # Debugging print
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("Input shape for CNN:", input_shape)

    # Definiuj klasyfikator CNN
    print("Defining CNN classifier")
    cnn_classifier = KerasClassifier(
        model=lambda: create_cnn_model(input_shape, num_classes),
        epochs=10,
        batch_size=32,
        verbose=0,
    )
    print("Defined CNN classifier")

    # Definiuj WCNNE z AdaBoost
    print("Defining AdaBoost model")
    model = AdaBoostClassifier(
        cnn_classifier,  # base_estimator is the first argument
        n_estimators=3,  # 3 CNN w WCNNE, jak w artykule
        learning_rate=0.01,
        random_state=42,
    )
    print("Defined AdaBoost model")

    # Trenuj model
    print("Training AdaBoost model")
    model.fit(X_train_scaled, y_train)
    print("Trained AdaBoost model")

    print("Exiting metoda6")
    return model, X_test, y_test