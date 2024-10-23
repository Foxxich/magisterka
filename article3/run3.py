import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# Load datasets
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Combine datasets
fake_news['label'] = 0
true_news['label'] = 1
data = pd.concat([fake_news, true_news])

# Data preprocessing (basic)
X = data['text']
y = data['label']

# Vectorization using CountVectorizer (can also use TfidfVectorizer)
vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Define classifiers
svm = SVC(probability=True)
nb = MultinomialNB()
dt = DecisionTreeClassifier()
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

# Voting classifier
voting_clf = VotingClassifier(estimators=[
    ('svm', svm), 
    ('nb', nb), 
    ('dt', dt), 
    ('bagging', bagging), 
    ('adaboost', adaboost)], voting='hard')

# Train voting classifier
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {accuracy * 100:.2f}%")
