import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2

# Load the datasets
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Label the datasets
fake_news['label'] = 0
true_news['label'] = 1

# Combine the datasets
news = pd.concat([fake_news, true_news]).reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(news['text'], news['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Feature selection (Chi-square)
selector = SelectKBest(chi2, k=5000)
X_train_tfidf = selector.fit_transform(X_train_tfidf, y_train)
X_test_tfidf = selector.transform(X_test_tfidf)

# Classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
adb = AdaBoostClassifier(n_estimators=100, random_state=42)

# Ensemble Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', rf), 
    ('lr', lr), 
    ('adb', adb)
], voting='hard')

# Train the ensemble model
voting_clf.fit(X_train_tfidf, y_train)

# Predictions
y_pred = voting_clf.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
