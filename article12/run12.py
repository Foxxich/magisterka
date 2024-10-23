# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Adding a 'label' column: 1 for fake news, 0 for real news
fake_news['label'] = 1
true_news['label'] = 0

# Combine the datasets
news_data = pd.concat([fake_news, true_news], ignore_index=True)

# Data Preprocessing: Cleaning and Feature Extraction
# Combine title and text into one feature
news_data['content'] = news_data['title'] + " " + news_data['text']

# Define X (features) and y (labels)
X = news_data['content']
y = news_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Encoding using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Random Forest (Bagging) Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
rf_predictions = rf_classifier.predict(X_test_tfidf)

# CatBoost (Boosting) Model
catboost_classifier = CatBoostClassifier(iterations=200, learning_rate=0.01, eval_metric='Accuracy', 
                                         early_stopping_rounds=20, use_best_model=True, verbose=50)
catboost_classifier.fit(X_train_tfidf, y_train, eval_set=(X_test_tfidf, y_test), plot=False)
catboost_predictions = catboost_classifier.predict(X_test_tfidf)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    print(f"Performance of {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("-" * 50)

# Evaluate Random Forest model
evaluate_model(y_test, rf_predictions, "Random Forest")

# Evaluate CatBoost model
evaluate_model(y_test, catboost_predictions, "CatBoost")
