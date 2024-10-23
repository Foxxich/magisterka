import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add labels
fake_news['label'] = 0
true_news['label'] = 1

# Combine datasets
news = pd.concat([fake_news, true_news])

# Preprocessing
X = news['text']
y = news['label']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Define base classifiers
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(kernel='linear', probability=True)

# Define ensemble methods
bagging = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=50, random_state=42)
boosting = AdaBoostClassifier(n_estimators=100, random_state=42)
voting = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('svc', svc)], voting='soft')

# Train and evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

print("Random Forest:")
evaluate_model(rf, X_train, X_test, y_train, y_test)

print("Bagging:")#too long
evaluate_model(bagging, X_train, X_test, y_train, y_test)

print("Boosting (AdaBoost):")
evaluate_model(boosting, X_train, X_test, y_train, y_test)

print("Voting Classifier:")
evaluate_model(voting, X_train, X_test, y_train, y_test)

# PS C:\Users\Vadym\Documents\magisterka\article9> python run9.py
# Random Forest:
# Accuracy: 0.9881959910913141
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      7091
#            1       0.99      0.99      0.99      6379

#     accuracy                           0.99     13470
#    macro avg       0.99      0.99      0.99     13470
# weighted avg       0.99      0.99      0.99     13470

# Bagging:
# Accuracy: 0.9888641425389755
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      7091
#            1       0.99      0.99      0.99      6379

#     accuracy                           0.99     13470
# weighted avg       0.99      0.99      0.99     13470

# Boosting (AdaBoost):
# C:\Users\Vadym\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\ensemble\_weight_boosting.py:527: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
#   warnings.warn(
# Accuracy: 0.9955456570155902
#            1       0.99      1.00      1.00      6379

#     accuracy                           1.00     13470
#    macro avg       1.00      1.00      1.00     13470
# weighted avg       1.00      1.00      1.00     13470

# Voting Classifier:
# Accuracy: 0.9926503340757238
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      7091
#            1       0.99      0.99      0.99      6379

#     accuracy                           0.99     13470
#    macro avg       0.99      0.99      0.99     13470
# weighted avg       0.99      0.99      0.99     13470

# PS C:\Users\Vadym\Documents\magisterka\article9> \