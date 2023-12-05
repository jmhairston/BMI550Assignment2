#JHairston Assignment 2

import warnings
warnings.filterwarnings("ignore")

####################################

import pandas as pd

# Load datasets
train_data = pd.read_csv('/Users/jmhairston/Desktop/Education/PhD/Fall 2023/BMI 550/fallreports_2023-9-21_train.csv')
test_data = pd.read_csv('/Users/jmhairston/Desktop/Education/PhD/Fall 2023/BMI 550/fallreports_2023-9-21_test.csv')

X_train, y_train = train_data['fall_description'], train_data['fog_q_class']
X_test, y_test = test_data['fall_description'], test_data['fog_q_class']

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def generate_features(X_train_text, X_test_text):
    #Preprocess
    X_train_text = np.where(pd.isnull(X_train_text), '', X_train_text)
    X_test_text = np.where(pd.isnull(X_test_text), '', X_test_text)

    # TF-IDF for n-grams
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    # Clustering-based features
    kmeans = KMeans(n_clusters=100, random_state=42)
    kmeans.fit(X_train_tfidf)
    X_train_cluster_labels = kmeans.predict(X_train_tfidf)
    X_test_cluster_labels = kmeans.predict(X_test_tfidf)

    # Combine features into a single feature set
    X_train_combined = np.concatenate((X_train_tfidf.toarray(), X_train_cluster_labels.reshape(-1, 1)), axis=1)
    X_test_combined = np.concatenate((X_test_tfidf.toarray(), X_test_cluster_labels.reshape(-1, 1)), axis=1)
    
    return X_train_combined, X_test_combined

X_train_combined, X_test_combined = generate_features(X_train, X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB()
}

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

for name, model in models.items():
    model.fit(X_train_combined, y_train)
    y_pred = model.predict(X_test_combined)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred)  # Calculate AUC
    
    print(f"Evaluation Metrics for {name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Classification Report
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

'''
Evaluation Metrics for Logistic Regression:
Accuracy: 0.8169
F1 Micro: 0.8169
F1 Macro: 0.8156
Precision: 0.8230
Recall: 0.8169
AUC: 0.8159

Evaluation Metrics for Random Forest:
Accuracy: 0.7324
F1 Micro: 0.7324
F1 Macro: 0.7245
Precision: 0.7560
Recall: 0.7324
AUC: 0.7302

Evaluation Metrics for Gradient Boosting:
Accuracy: 0.7606
F1 Micro: 0.7606
F1 Macro: 0.7598
Precision: 0.7622
Recall: 0.7606
AUC: 0.7599

Evaluation Metrics for SVM:
Accuracy: 0.5070
F1 Micro: 0.5070
F1 Macro: 0.3604
Precision: 0.7535
Recall: 0.5070
AUC: 0.5139

Evaluation Metrics for Naive Bayes:
Accuracy: 0.6479
F1 Micro: 0.6479
F1 Macro: 0.6025
Precision: 0.7527
Recall: 0.6479
AUC: 0.6433
'''

results = {
    'Logistic Regression': {
        'accuracy': 0.8169,
        'f1_micro': 0.8169,
        'f1_macro': 0.8156,
        'precision': 0.8230,
        'recall': 0.8169,
        'auc': 0.8159
    },
    'Random Forest': {
        'accuracy': 0.7324,
        'f1_micro': 0.7324,
        'f1_macro': 0.7245,
        'precision': 0.7560,
        'recall': 0.7324,
        'auc': 0.7302
    },
    'XGB': {
        'accuracy': 0.7606,
        'f1_micro': 0.7606,
        'f1_macro': 0.7598,
        'precision': 0.7622,
        'recall': 0.7606,
        'auc': 0.7599
    },
    'SVM': {
        'accuracy': 0.5070,
        'f1_micro': 0.5070,
        'f1_macro': 0.3604,
        'precision': 0.7535,
        'recall': 0.5070,
        'auc': 0.5139
    },
    'Naive Bayes': {
        'accuracy': 0.6479,
        'f1_micro': 0.6479,
        'f1_macro': 0.6025,
        'precision': 0.7527,
        'recall': 0.6479,
        'auc': 0.6433
    }
}

#Based on overall micro f1-score results the best perfoming classifer is the Log Regression

# Training vs Performance

import matplotlib.pyplot as plt

best_classifier = max(results, key=lambda x: results[x]['f1_micro'])

# Initialize the best performing classifier
best_model = models[best_classifier]

# Varying training set sizes
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]  # Proportions of the total dataset

# Lists to store performance metrics
f1_scores_micro = []

# Iterate over different training set sizes
for size in train_sizes:
    # Calculate the number of samples for the current size
    num_samples = int(size * len(X_train_combined))
    
    # Select a subset of the training data
    X_subset = X_train_combined[:num_samples]
    y_subset = y_train[:num_samples]
    
    # Train the model on the subset
    best_model.fit(X_subset, y_subset)
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test_combined)
    
    # Calculate f1_micro and store in the list
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_scores_micro.append(f1_micro)

# Plotting training set size vs. accuracy
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, f1_scores_micro, marker='o')
plt.title('Training Set Size vs. Micro f1 for the Best Performing Classifier')
plt.xlabel('Training Set Size')
plt.ylabel('F1_Micro')
plt.xticks(train_sizes)
plt.grid(True)
plt.tight_layout()
plt.show()

#Ablation Study

best_model = models[best_classifier]
f1_scores = []

num_features = X_train_combined.shape[1]

for feature_index in range(num_features):
    # Copy of feature set with one feature removed
    X_train_ablated = np.delete(X_train_combined, feature_index, axis=1)
    X_test_ablated = np.delete(X_test_combined, feature_index, axis=1)
    
    best_model.fit(X_train_ablated, y_train)
    y_pred = best_model.predict(X_test_ablated)
    
    # Calculate F1 score and store in the list
    f1 = f1_score(y_test, y_pred, average='micro')
    f1_scores.append(f1)

# Plotting feature ablation study results
plt.figure(figsize=(8, 6))
plt.plot(range(num_features), f1_scores, marker='o')
plt.title('Feature Ablation Study for Best Performing Classifier')
plt.xlabel('Feature Index')
plt.ylabel('Micro F1 Score')
plt.xticks(range(num_features))
plt.grid(True)
plt.tight_layout()
plt.show()
