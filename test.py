import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Load dataset
column_names = [
    "Class", "Alcohol", "Malic_Acid", "Ash", "Alkalinity_of_Ash",
    "Magnesium", "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280/OD315", "Proline"
]
data = pd.read_csv("wine/wine.data", header=None, names=column_names)

# Generate synthetic textual descriptions
texts = [
    f"Alcohol: {row['Alcohol']:.1f}, Malic Acid: {row['Malic_Acid']:.1f}, Phenols: {row['Total_Phenols']:.1f}, "
    f"Color Intensity: {row['Color_Intensity']:.1f}"
    for _, row in data.iterrows()
]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(texts)

# Train-test split for text data
text_X_train, text_X_test, text_y_train, text_y_test = train_test_split(
    text_vectors, data['Class'], test_size=0.3, random_state=42
)

# Define Models for Text Analysis
text_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "kNN": KNeighborsClassifier()
}

# Train and Evaluate Text Models
text_accuracies = []
for name, model in text_models.items():
    # Fit the model
    model.fit(text_X_train, text_y_train)
    
    # Predictions
    text_y_train_pred = model.predict(text_X_train)
    text_y_test_pred = model.predict(text_X_test)
    
    # Accuracy
    train_accuracy = accuracy_score(text_y_train, text_y_train_pred)
    test_accuracy = accuracy_score(text_y_test, text_y_test_pred)
    text_accuracies.append((name, train_accuracy, test_accuracy))
    
    # Print Classification Reports
    print(f"=== Classification Report for {name} (Training Data) ===")
    print(classification_report(text_y_train, text_y_train_pred, zero_division=0))
    print(f"=== Classification Report for {name} (Testing Data) ===")
    print(classification_report(text_y_test, text_y_test_pred, zero_division=0))
    print("-" * 80)
    
    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, text_X_test, text_y_test, cmap="viridis")
    plt.title(f"Confusion Matrix - {name} (Text Analysis)")
    plt.show()

# Accuracy Comparison for Text Analysis
models, train_acc, test_acc = zip(*text_accuracies)
plt.figure(figsize=(12, 6))

# Bar plot for testing accuracy
sns.barplot(x=list(models), y=list(test_acc), palette=sns.color_palette("viridis", len(models)))
plt.title("Text Analysis Model Accuracy Comparison (Testing Data)")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)

# Annotate barplot with accuracy values
for i, v in enumerate(test_acc):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')

plt.show()

# Bar plot for training accuracy
plt.figure(figsize=(12, 6))
sns.barplot(x=list(models), y=list(train_acc), palette=sns.color_palette("mako", len(models)))
plt.title("Text Analysis Model Accuracy Comparison (Training Data)")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)

# Annotate barplot with accuracy values
for i, v in enumerate(train_acc):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')

plt.show()
