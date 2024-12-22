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
data = pd.read_csv("wine\wine.data", header=None, names=column_names)

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
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracies.append((name, accuracy_score(y_test, y_pred)))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap="viridis")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Accuracy Comparison
names, acc = zip(*accuracies)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(names), y=list(acc), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.show()

# Text Analysis
# Generate synthetic textual descriptions
texts = [
    f"Alcohol: {row['Alcohol']:.1f}, Acid: {row['Malic_Acid']:.1f}, Phenols: {row['Total_Phenols']:.1f}" 
    for _, row in data.iterrows()
]

vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(texts)
text_X_train, text_X_test, text_y_train, text_y_test = train_test_split(
    text_vectors, y, test_size=0.3, random_state=42
)

text_model = LogisticRegression(max_iter=1000, random_state=42)
text_model.fit(text_X_train, text_y_train)
text_y_pred = text_model.predict(text_X_test)

print("Text Analysis Accuracy:", accuracy_score(text_y_test, text_y_pred))
print("Classification Report:\n", classification_report(text_y_test, text_y_pred))

# Confusion Matrix for Text Analysis
disp = ConfusionMatrixDisplay.from_estimator(text_model, text_X_test, text_y_test, cmap="viridis")
plt.title("Confusion Matrix - Text Analysis")
plt.show()

# Feature Importance Visualization (Top 10 Features)
feature_names = vectorizer.get_feature_names_out()
coefficients = text_model.coef_[0]
top_features = sorted(zip(coefficients, feature_names), key=lambda x: abs(x[0]), reverse=True)[:10]
weights, features = zip(*top_features)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(weights), y=list(features), palette="mako")
plt.title("Top 10 Features - Logistic Regression (Text Analysis)")
plt.xlabel("Feature Coefficient")
plt.ylabel("Feature")
plt.show()