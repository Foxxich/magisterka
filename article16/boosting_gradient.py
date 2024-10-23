
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load ISOT dataset
fake_news = pd.read_csv(r"C:\Users\Vadym\Documents\magisterka\datasets\ISOT_dataset\Fake.csv")
true_news = pd.read_csv(r"C:\Users\Vadym\Documents\magisterka\datasets\ISOT_dataset\True.csv")

# Add labels
fake_news['label'] = 0
true_news['label'] = 1

# Combine datasets
news_data = pd.concat([fake_news, true_news])

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(news_data['text'], news_data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Gradient Boosting Classifier
boosting_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
boosting_clf.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred_boosting = boosting_clf.predict(X_test_tfidf)
boosting_accuracy = accuracy_score(y_test, y_pred_boosting)
print(f"Boosting Classifier Accuracy: {boosting_accuracy * 100:.2f}%")
