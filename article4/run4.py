import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional, concatenate
from keras.optimizers import Adadelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the ISOT Dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add labels: 0 for fake, 1 for true
fake_news['label'] = 0
true_news['label'] = 1

# Combine datasets
data = pd.concat([fake_news, true_news], ignore_index=True)

# Preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove special characters
    return text

data['text'] = data['text'].apply(preprocess_text)

# Tokenization and Padding
MAX_NB_WORDS = 20000  # Vocabulary size
MAX_SEQUENCE_LENGTH = 300  # Max length of each news item
EMBEDDING_DIM = 100  # Embedding dimension for word vectors

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
word_index = tokenizer.word_index

X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label encoding for classification
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Bi-LSTM Model
input_lstm = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_lstm = Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_lstm)
bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_lstm)
flatten_lstm = Flatten()(bi_lstm)

# CNN Model
input_cnn = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_cnn = Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_cnn)
conv = Conv1D(128, 5, activation='relu')(embedding_cnn)
maxpool = MaxPooling1D(pool_size=5)(conv)
flatten_cnn = Flatten()(maxpool)

# Combine CNN and LSTM outputs
merged = concatenate([flatten_lstm, flatten_cnn])

# Fully connected layer
dense_1 = Dense(128, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense_1)

# Final Model
model = Model(inputs=[input_lstm, input_cnn], outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

# Train the model with fewer epochs and larger batch size
model.fit([X_train, X_train], y_train, epochs=3, batch_size=256, validation_data=([X_test, X_test], y_test))
# Evaluate the model
score = model.evaluate([X_test, X_test], y_test)
print(f'Test accuracy: {score[1]}')

