import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add labels to datasets
fake_news['label'] = 0  # Fake news
true_news['label'] = 1  # True news

# Combine datasets
news_data = pd.concat([fake_news, true_news], axis=0).reset_index(drop=True)

# Preprocess the text data
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove single characters and numbers
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^\s+|\s+?$', '', text.lower())
    return text

# Initialize a stemmer
snowball_stemmer = SnowballStemmer('english')

# Apply preprocessing and stemming
news_data['text'] = news_data['text'].apply(preprocess_text)
news_data['text'] = news_data['text'].apply(lambda x: ' '.join([snowball_stemmer.stem(word) for word in x.split()]))

# Convert the text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(news_data['text']).toarray()
y = news_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models for stacking
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(max_iter=200)

# Create a stacking classifier
stack_model = StackingClassifier(estimators=[('rf', rf_model), ('ab', ab_model)], final_estimator=lr_model)

# Train the stacking model
stack_model.fit(X_train, y_train)

# Make predictions
y_pred = stack_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
