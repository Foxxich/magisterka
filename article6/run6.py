import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np

# Load the dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add a label to each dataset
fake_news['label'] = 0
true_news['label'] = 1

# Combine the datasets
news_data = pd.concat([fake_news, true_news]).sample(frac=1).reset_index(drop=True)

# Preprocess the text (e.g., remove symbols, etc.)
def clean_text(text):
    # Remove unwanted characters, symbols, etc.
    return text.str.replace(r'[^\w\s]', '', regex=True).str.lower()

news_data['text'] = clean_text(news_data['text'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(news_data['text'], news_data['label'], test_size=0.2, random_state=42)

# Use BERT tokenizer and model for text embedding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def bert_encode(texts, tokenizer, max_len=128):
    tokens = tokenizer(texts.tolist(), max_length=max_len, truncation=True, padding=True, return_tensors='tf')
    return tokens

X_train_encoded = bert_encode(X_train, tokenizer)
X_test_encoded = bert_encode(X_test, tokenizer)

# Build a CNN model for AdaBoost ensemble learning
def build_cnn_model():
    input_ids = tf.keras.Input(shape=(128,), dtype='int32', name='input_ids')
    attention_mask = tf.keras.Input(shape=(128,), dtype='int32', name='attention_mask')
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]  # Take pooled output
    cnn = tf.keras.layers.Conv1D(128, 5, activation='relu')(tf.expand_dims(bert_output, -1))
    cnn = tf.keras.layers.GlobalMaxPooling1D()(cnn)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(cnn)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()

# Train the CNN model
cnn_model.fit({'input_ids': X_train_encoded['input_ids'], 'attention_mask': X_train_encoded['attention_mask']},
              y_train, epochs=3, batch_size=32, validation_split=0.2)

# Use AdaBoost to create an ensemble of CNN models
adaboost_model = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=50)

# Convert the CNN output into features for AdaBoost
X_train_cnn_features = cnn_model.predict({'input_ids': X_train_encoded['input_ids'], 'attention_mask': X_train_encoded['attention_mask']}).flatten()
X_test_cnn_features = cnn_model.predict({'input_ids': X_test_encoded['input_ids'], 'attention_mask': X_test_encoded['attention_mask']}).flatten()

# Fit AdaBoost on the CNN features
adaboost_model.fit(X_train_cnn_features.reshape(-1, 1), y_train)

# Evaluate the model
y_pred = adaboost_model.predict(X_test_cnn_features.reshape(-1, 1))
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}, F1-Score: {f1}, Precision: {precision}, Recall: {recall}")
