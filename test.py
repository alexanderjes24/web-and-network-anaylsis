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

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
ConfusionMatrixDisplay.from_estimator(logistic_model, X_test_scaled, y_test, cmap="viridis")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_scaled, y_train)
y_pred_tree = tree_model.predict(X_test_scaled)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
ConfusionMatrixDisplay.from_estimator(tree_model, X_test_scaled, y_test, cmap="viridis")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Random Forest
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train_scaled, y_train)
y_pred_forest = forest_model.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_forest))
ConfusionMatrixDisplay.from_estimator(forest_model, X_test_scaled, y_test, cmap="viridis")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
ConfusionMatrixDisplay.from_estimator(svm_model, X_test_scaled, y_test, cmap="viridis")
plt.title("Confusion Matrix - SVM")
plt.show()

# k-Nearest Neighbors (kNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
print("kNN Accuracy:", accuracy_score(y_test, y_pred_knn))
ConfusionMatrixDisplay.from_estimator(knn_model, X_test_scaled, y_test, cmap="viridis")
plt.title("Confusion Matrix - kNN")
plt.show()

accuracies = []
names, acc = zip(*accuracies)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=list(names), y=list(acc), palette=sns.color_palette("viridis", len(names)))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
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

# Logistic Regression for Text Analysis
text_logistic_model = LogisticRegression(max_iter=1000, random_state=42)
text_logistic_model.fit(text_X_train, text_y_train)
text_y_pred_logistic = text_logistic_model.predict(text_X_test)
print("Text Analysis - Logistic Regression Accuracy:", accuracy_score(text_y_test, text_y_pred_logistic))
ConfusionMatrixDisplay.from_estimator(text_logistic_model, text_X_test, text_y_test, cmap="viridis")
plt.title("Confusion Matrix - Logistic Regression (Text Analysis)")
plt.show()

# Decision Tree for Text Analysis
text_tree_model = DecisionTreeClassifier(random_state=42)
text_tree_model.fit(text_X_train, text_y_train)
text_y_pred_tree = text_tree_model.predict(text_X_test)
print("Text Analysis - Decision Tree Accuracy:", accuracy_score(text_y_test, text_y_pred_tree))
ConfusionMatrixDisplay.from_estimator(text_tree_model, text_X_test, text_y_test, cmap="viridis")
plt.title("Confusion Matrix - Decision Tree (Text Analysis)")
plt.show()

# Random Forest for Text Analysis
text_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
text_forest_model.fit(text_X_train, text_y_train)
text_y_pred_forest = text_forest_model.predict(text_X_test)
print("Text Analysis - Random Forest Accuracy:", accuracy_score(text_y_test, text_y_pred_forest))
ConfusionMatrixDisplay.from_estimator(text_forest_model, text_X_test, text_y_test, cmap="viridis")
plt.title("Confusion Matrix - Random Forest (Text Analysis)")
plt.show()

# Support Vector Machine (SVM) for Text Analysis
text_svm_model = SVC(random_state=42)
text_svm_model.fit(text_X_train, text_y_train)
text_y_pred_svm = text_svm_model.predict(text_X_test)
print("Text Analysis - SVM Accuracy:", accuracy_score(text_y_test, text_y_pred_svm))
ConfusionMatrixDisplay.from_estimator(text_svm_model, text_X_test, text_y_test, cmap="viridis")
plt.title("Confusion Matrix - SVM (Text Analysis)")
plt.show()

# k-Nearest Neighbors (kNN) for Text Analysis
text_knn_model = KNeighborsClassifier()
text_knn_model.fit(text_X_train, text_y_train)
text_y_pred_knn = text_knn_model.predict(text_X_test)
print("Text Analysis - kNN Accuracy:", accuracy_score(text_y_test, text_y_pred_knn))
ConfusionMatrixDisplay.from_estimator(text_knn_model, text_X_test, text_y_test, cmap="viridis")
plt.title("Confusion Matrix - kNN (Text Analysis)")
plt.show()
