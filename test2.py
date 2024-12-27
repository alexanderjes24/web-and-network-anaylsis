import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
tree_model = DecisionTreeClassifier(random_state=42)
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(random_state=42)
knn_model = KNeighborsClassifier()

# Train and evaluate Logistic Regression
logistic_model.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_model.predict(X_test_scaled)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_report = classification_report(y_test, y_pred_logistic, output_dict=True)

# Train and evaluate Decision Tree
tree_model.fit(X_train_scaled, y_train)
y_pred_tree = tree_model.predict(X_test_scaled)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_report = classification_report(y_test, y_pred_tree, output_dict=True)

# Train and evaluate Random Forest
forest_model.fit(X_train_scaled, y_train)
y_pred_forest = forest_model.predict(X_test_scaled)
forest_accuracy = accuracy_score(y_test, y_pred_forest)
forest_report = classification_report(y_test, y_pred_forest, output_dict=True)

# Train and evaluate SVM
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)

# Train and evaluate k-Nearest Neighbors
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_report = classification_report(y_test, y_pred_knn, output_dict=True)

# Create a summary table for results
results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "kNN"],
    "Accuracy": [logistic_accuracy, tree_accuracy, forest_accuracy, svm_accuracy, knn_accuracy],
    "Precision": [
        logistic_report["weighted avg"]["precision"],
        tree_report["weighted avg"]["precision"],
        forest_report["weighted avg"]["precision"],
        svm_report["weighted avg"]["precision"],
        knn_report["weighted avg"]["precision"],
    ],
    "Recall": [
        logistic_report["weighted avg"]["recall"],
        tree_report["weighted avg"]["recall"],
        forest_report["weighted avg"]["recall"],
        svm_report["weighted avg"]["recall"],
        knn_report["weighted avg"]["recall"],
    ],
    "F1-Score": [
        logistic_report["weighted avg"]["f1-score"],
        tree_report["weighted avg"]["f1-score"],
        forest_report["weighted avg"]["f1-score"],
        svm_report["weighted avg"]["f1-score"],
        knn_report["weighted avg"]["f1-score"],
    ],
})

# Print the table in the terminal
print("\n=== Model Performance Summary ===")
print(results_df.to_string(index=False))

# Accuracy Comparison Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df["Model"], y=results_df["Accuracy"], palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)

# Annotate each bar with its accuracy value
for i, acc in enumerate(results_df["Accuracy"]):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=10, fontweight="bold")

plt.show()
