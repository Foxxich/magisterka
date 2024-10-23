import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Embedding, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling1D

# Load datasets
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Preprocessing text data
def preprocess_text(df):
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)  # Remove punctuation
    df['text'] = df['text'].str.replace('\d+', '', regex=True)  # Remove numbers
    return df

fake_news = preprocess_text(fake_news)
true_news = preprocess_text(true_news)

# Label the datasets
fake_news['label'] = 0
true_news['label'] = 1

# Combine datasets
news_data = pd.concat([fake_news, true_news])

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(news_data['text'])
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(news_data['text'])
maxlen = 300
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Train/test split for text data
X_train_text, X_test_text, y_train, y_test = train_test_split(padded_sequences, news_data['label'], test_size=0.2, random_state=42)

# Prepare embeddings using GloVe (random embeddings used for demonstration)
embedding_dim = 100
embedding_matrix = np.random.rand(vocab_size, embedding_dim)

# Text feature extractor using Bi-LSTM
input_text = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)(input_text)
lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
lstm_output = GlobalAveragePooling1D()(lstm_layer)

# Final classification layer
dense_1 = Dense(128, activation='relu')(lstm_output)
dropout = Dropout(0.5)(dense_1)
output = Dense(1, activation='sigmoid')(dropout)

# Build and compile model
model = Model(inputs=input_text, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train_text, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test_text)
y_pred = (y_pred > 0.5).astype(int)

# Performance metrics
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig("res.jpg")
