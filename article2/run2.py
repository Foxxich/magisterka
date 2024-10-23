import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load the ISOT dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add a label column
fake_news['label'] = 0
true_news['label'] = 1

# Combine the datasets
data = pd.concat([fake_news, true_news])

# Split into features and labels
X = data['text']  # Assuming the text column is named 'text'
y = data['label']

# Convert text to numerical features using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')  # Customize max_features as needed
X_tfidf = tfidf.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.35, random_state=42)

# 1. Train the Boosted Decision Tree on the transformed text features
boosted_tree = GradientBoostingClassifier()
boosted_tree.fit(X_train, y_train)
boosted_tree_preds = boosted_tree.predict_proba(X_test)[:, 1]  # Get the probability estimates

# 2. Train the Neural Network on the same features
neural_net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
neural_net.fit(X_train, y_train)
neural_net_preds = neural_net.predict_proba(X_test)[:, 1]

# 3. Combine predictions from both models
combined_preds = pd.DataFrame({
    'boosted_tree': boosted_tree_preds,
    'neural_net': neural_net_preds
})

# 4. Train Logistic Regression as the gating model
logistic_reg = LogisticRegression()
logistic_reg.fit(combined_preds, y_test)
final_preds = logistic_reg.predict(combined_preds)

# Evaluation
print("Accuracy:", accuracy_score(y_test, final_preds))
print(classification_report(y_test, final_preds))
