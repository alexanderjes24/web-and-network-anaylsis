import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Preprocessing
X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "kNN": KNeighborsClassifier()
}

accuracies = []
for name, model in models.items():
    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Predict on both training and testing sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Accuracy calculation
    test_accuracy = accuracy_score(y_test, y_test_pred)
    accuracies.append((name, test_accuracy))
    print(f"=== Classification Report for {name} (Training Data) ===")
    print(classification_report(y_train, y_train_pred, zero_division=0))
    print(f"=== Classification Report for {name} (Testing Data) ===")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    print("-" * 80)
    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap="viridis")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Accuracy Comparison
names, acc = zip(*accuracies)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=list(names), y=list(acc), palette=sns.color_palette("viridis", len(names)))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)

# Annotate each bar with its accuracy value
for i, v in enumerate(acc):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')

plt.show()


# ---- Text Analysis for All Models ----

# Create synthetic textual descriptions
texts = [
    f"Alcohol: {row['Alcohol']:.1f}, Malic Acid: {row['Malic_Acid']:.1f}, Phenols: {row['Total_Phenols']:.1f}, "
    f"Color Intensity: {row['Color_Intensity']:.1f}"
    for _, row in data.iterrows()
]

vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(texts)
text_X_train, text_X_test, text_y_train, text_y_test = train_test_split(
    text_vectors, y, test_size=0.3, random_state=42
)

# Dictionary of text models
text_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "kNN": KNeighborsClassifier()
}

# Train and evaluate each model on text data
for name, model in text_models.items():
    # Fit the model
    model.fit(text_X_train, text_y_train)

    # Predict on test set
    text_y_pred = model.predict(text_X_test)

    # Accuracy calculation
    print(f"=== Classification Report for {name} (Text Analysis) ===")
    print(classification_report(text_y_test, text_y_pred, zero_division=0))

    # Confusion Matrix for Text Analysis
    disp = ConfusionMatrixDisplay.from_estimator(model, text_X_test, text_y_test, cmap="viridis")
    plt.title(f"Confusion Matrix - {name} (Text Analysis)")
    plt.show()

    # Feature Importance (only applicable to models that can output feature importances, e.g., RandomForest)
    if hasattr(model, 'coef_'):  # Models like Logistic Regression have coefficients
        coefficients = np.mean(model.coef_, axis=0)
        top_features = sorted(zip(coefficients, vectorizer.get_feature_names_out()), key=lambda x: abs(x[0]), reverse=True)[:10]
        weights, features = zip(*top_features)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(weights), y=list(features), palette="mako")
        plt.title(f"Top 10 Features - {name} (Text Analysis)")
        plt.xlabel("Feature Coefficient")
        plt.ylabel("Feature")
        plt.show()
    
    elif hasattr(model, 'feature_importances_'):  # Models like RandomForest have feature importances
        importances = model.feature_importances_
        top_features = sorted(zip(importances, vectorizer.get_feature_names_out()), key=lambda x: x[0], reverse=True)[:10]
        weights, features = zip(*top_features)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(weights), y=list(features), palette="mako")
        plt.title(f"Top 10 Features - {name} (Text Analysis)")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.show()
