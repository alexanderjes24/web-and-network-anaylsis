# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sentence_transformers import SentenceTransformer

# Load dataset
column_names = [
    "Class", "Alcohol", "Malic_Acid", "Ash", "Alkalinity_of_Ash",
    "Magnesium", "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280/OD315", "Proline"
]
data = pd.read_csv("wine/wine.data", header=None, names=column_names)

# Generate enhanced synthetic textual descriptions
texts = [
    f"The wine has an alcohol content of {row['Alcohol']:.1f}, malic acid level of {row['Malic_Acid']:.1f}, "
    f"a total phenols measure of {row['Total_Phenols']:.1f}, and a color intensity rated at {row['Color_Intensity']:.1f}."
    for _, row in data.iterrows()
]

# Split data into train and test sets
text_X_train, text_X_test, text_y_train, text_y_test = train_test_split(
    texts, data['Class'], test_size=0.3, random_state=42
)

# === Approach 1: TF-IDF Vectorization ===
# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_X_train = tfidf_vectorizer.fit_transform(text_X_train)
tfidf_X_test = tfidf_vectorizer.transform(text_X_test)

# Models to test
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (Linear Kernel)": SVC(kernel='linear', random_state=42)
}

# Train and evaluate models using TF-IDF
tfidf_accuracies = []
for name, model in models.items():
    model.fit(tfidf_X_train, text_y_train)
    y_train_pred = model.predict(tfidf_X_train)
    y_test_pred = model.predict(tfidf_X_test)

    # Accuracy scores
    train_accuracy = accuracy_score(text_y_train, y_train_pred)
    test_accuracy = accuracy_score(text_y_test, y_test_pred)
    tfidf_accuracies.append((name, train_accuracy, test_accuracy))

    # Classification report and confusion matrix
    print(f"=== Classification Report for {name} (TF-IDF) ===")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print(classification_report(text_y_test, y_test_pred, zero_division=0))
    ConfusionMatrixDisplay.from_estimator(model, tfidf_X_test, text_y_test, cmap="viridis")
    plt.title(f"Confusion Matrix - {name} (TF-IDF)")
    plt.show()

# === Approach 2: Pre-trained Sentence Embeddings ===
# Load pre-trained model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_X_train = embedding_model.encode(text_X_train)
embedding_X_test = embedding_model.encode(text_X_test)

# Train and evaluate models using Sentence Embeddings
embedding_accuracies = []
for name, model in models.items():
    model.fit(embedding_X_train, text_y_train)
    y_train_pred = model.predict(embedding_X_train)
    y_test_pred = model.predict(embedding_X_test)

    # Accuracy scores
    train_accuracy = accuracy_score(text_y_train, y_train_pred)
    test_accuracy = accuracy_score(text_y_test, y_test_pred)
    embedding_accuracies.append((name, train_accuracy, test_accuracy))

    # Classification report and confusion matrix
    print(f"=== Classification Report for {name} (Embeddings) ===")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    print(classification_report(text_y_test, y_test_pred, zero_division=0))
    ConfusionMatrixDisplay.from_estimator(model, embedding_X_test, text_y_test, cmap="plasma")
    plt.title(f"Confusion Matrix - {name} (Embeddings)")
    plt.show()

# === Accuracy Comparison ===
# Combine accuracies
combined_accuracies = {
    "TF-IDF": tfidf_accuracies,
    "Embeddings": embedding_accuracies
}

for method, accuracies in combined_accuracies.items():
    models, train_acc, test_acc = zip(*accuracies)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(models), y=list(test_acc), palette="viridis")
    plt.title(f"{method} Model Accuracy Comparison (Testing Data)")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    for i, v in enumerate(test_acc):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')
    plt.show()
