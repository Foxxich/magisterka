from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from common import load_shuffle_preprocess_data, split_data

def train_run8():
    # Wczytaj i przetwórz dane
    X, y = load_shuffle_preprocess_data()

    # Podziel dane
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Wektoryzacja (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Stwórz modele bazowe: RandomForest i XGBoost
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Połącz modele w VotingClassifier (metoda zespołowa)
    ensemble_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')

    # Wytrenuj model zespołowy
    ensemble_model.fit(X_train_tfidf, y_train)

    return ensemble_model, X_test_tfidf, y_test
