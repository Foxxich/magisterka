import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the ISOT dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Combine and label the datasets
fake_news['label'] = 0  # Fake news labeled as 0
true_news['label'] = 1  # True news labeled as 1

# Combine the datasets into one dataframe
data = pd.concat([fake_news, true_news]).reset_index(drop=True)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Split the dataset into features and labels
X = data['text']
y = data['label']

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Initialize the classifiers
mlp = MLPClassifier(alpha=0.01, hidden_layer_sizes=(14,), max_iter=100, solver='lbfgs', random_state=0)
log_reg = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=100, random_state=0)
xgb = XGBClassifier(gamma=1, learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=100, use_label_encoder=False, eval_metric='logloss')

# Create a voting classifier with soft voting
voting_clf = VotingClassifier(estimators=[('mlp', mlp), ('log_reg', log_reg), ('xgb', xgb)], voting='soft')

# Train the model
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])

# Print results
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"ROC AUC Score: {roc_auc*100:.2f}%")
