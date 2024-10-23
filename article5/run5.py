import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  # Progress bar library

# Load the ISOT dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add a label: 1 for fake news, 0 for true news
fake_news['label'] = 1
true_news['label'] = 0

# Combine the datasets
data = pd.concat([fake_news, true_news], axis=0)

# Preprocess and prepare the data
X = data['text']  # Features: the news articles
y = data['label']  # Labels: 1 for fake, 0 for true

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Progress bar for Random Forest Classifier training
print("Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Show progress using tqdm
for _ in tqdm(range(1), desc="Random Forest Fitting"):
    rf_clf.fit(X_train, y_train)

# Predict with Random Forest
rf_preds = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"Random Forest Test Accuracy: {rf_accuracy * 100:.2f}%")

# XGBoost Classifier training with progress bar
print("Training XGBoost...")
xgb_clf = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)

for _ in tqdm(range(1), desc="XGBoost Fitting"):
    xgb_clf.fit(X_train, y_train)

# Predict with XGBoost
xgb_preds = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f"XGBoost Test Accuracy: {xgb_accuracy * 100:.2f}%")

# Ensemble by averaging the predictions
final_preds = (rf_preds + xgb_preds) / 2
final_preds = np.round(final_preds)

# Evaluate the ensemble
ensemble_accuracy = accuracy_score(y_test, final_preds)
print(f"Ensemble Model Test Accuracy: {ensemble_accuracy * 100:.2f}%")
print(classification_report(y_test, final_preds))
