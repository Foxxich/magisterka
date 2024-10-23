import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the ISOT dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Labeling the data
fake_news['label'] = 0  # Fake news
true_news['label'] = 1  # True news

# Combine the datasets
news_data = pd.concat([fake_news, true_news])

# Preprocessing the text
X = news_data['text']
y = news_data['label']

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = xgb_model.predict(X_test_tfidf)

# Evaluate the performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
