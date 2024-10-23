import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load ISOT Fake News Dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add labels: Fake = 1, True = 0
fake_news['label'] = 1
true_news['label'] = 0

# Combine datasets
data = pd.concat([fake_news, true_news], axis=0).reset_index(drop=True)

# Preprocessing: Text and Labels
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF with a limited number of features (to reduce memory usage)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=50000)  # Reduce feature size
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Multinomial Naive Bayes (better suited for text classification)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)  # No need for dense matrix conversion

# Make predictions
y_pred_nb = naive_bayes.predict(X_test_tfidf)

# Evaluate the model
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
