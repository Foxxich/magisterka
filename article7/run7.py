# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import torch
import torch.nn.functional as F

# Load the ISOT Fake News Dataset
fake_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\Fake.csv")
true_news = pd.read_csv("C:\\Users\\Vadym\\Documents\\magisterka\\datasets\\ISOT_dataset\\True.csv")

# Add label column: 1 for real news, 0 for fake news
true_news['label'] = 1
fake_news['label'] = 0

# Concatenate the datasets
df = pd.concat([true_news, fake_news], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Split data into features (X) and target (y)
X = df['text']  # Assuming the news articles are in a column named 'text'
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


### Monte Carlo Dropout for Uncertainty Estimation ###

# Function for Monte Carlo Dropout Inference (Simulated here since RandomForest doesn't use dropout directly)
def mc_dropout_inference(model, X_test, num_samples=50):
    # Simulate the Monte Carlo by running multiple predictions (since RandomForest doesn't use dropout)
    predictions = []
    for _ in range(num_samples):
        preds = model.predict_proba(X_test)  # Get predicted probabilities
        predictions.append(preds)
    
    predictions = np.array(predictions)
    
    # Mean and variance of predictions
    mean_preds = predictions.mean(axis=0)
    var_preds = predictions.var(axis=0)
    
    return mean_preds, var_preds

# Run MC Dropout inference (simulated for RandomForest)
mean_predictions, uncertainty = mc_dropout_inference(rf_classifier, X_test_tfidf, num_samples=50)

# Get predicted labels from the mean predictions
pred_labels = np.argmax(mean_predictions, axis=1)

# Recalculate accuracy with MC Dropout predictions
mc_dropout_accuracy = accuracy_score(y_test, pred_labels)
print(f"Accuracy with Monte Carlo Dropout: {mc_dropout_accuracy}")

### Heuristic Post-Processing Based on Meta-Attributes ###

# Function to compute meta-attribute probabilities (Simulated for 'subject' column, if available)
def compute_meta_attribute_probabilities(df, attribute_col, label_col):
    attribute_probs = defaultdict(lambda: {'real': 0, 'fake': 0})
    
    for _, row in df.iterrows():
        attr_value = row[attribute_col]
        label = row[label_col]

        if label == 1:  # 'real' news
            attribute_probs[attr_value]['real'] += 1
        else:  # 'fake' news
            attribute_probs[attr_value]['fake'] += 1

    # Normalize probabilities
    for attr_value, counts in attribute_probs.items():
        total = counts['real'] + counts['fake']
        attribute_probs[attr_value]['real'] /= total
        attribute_probs[attr_value]['fake'] /= total

    return attribute_probs

# Apply heuristic correction
def heuristic_post_processing(predictions, attribute_probs, attribute_values, threshold=0.9):
    final_predictions = []
    
    for i, pred in enumerate(predictions):
        attr_value = attribute_values[i]  # Get the corresponding attribute value for this instance
        
        if attr_value in attribute_probs:
            real_prob = attribute_probs[attr_value]['real']
            fake_prob = attribute_probs[attr_value]['fake']
            
            # Heuristic rules based on probabilities
            if real_prob > threshold and real_prob > fake_prob:
                final_predictions.append(1)  # Real
            elif fake_prob > threshold and fake_prob > real_prob:
                final_predictions.append(0)  # Fake
            else:
                final_predictions.append(pred)  # Keep original prediction
        else:
            final_predictions.append(pred)  # If attribute not in training data, keep original prediction

    return final_predictions

# Simulate some meta-attribute (such as 'subject' or 'source')
df['source'] = df['subject'].apply(lambda x: x if x in ['politics', 'world', 'business'] else 'unknown')

# Calculate attribute probabilities based on 'source'
attribute_probs = compute_meta_attribute_probabilities(df, 'source', 'label')

# Apply heuristic post-processing
final_predictions = heuristic_post_processing(pred_labels, attribute_probs, df['source'], threshold=0.9)

# Evaluate final predictions after heuristic post-processing
final_accuracy = accuracy_score(y_test, final_predictions)
print(f"Accuracy after heuristic post-processing: {final_accuracy}")
