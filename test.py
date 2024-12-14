import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# Load the dataset
column_names = [
    'Class', 'Alcohol', 'Malic_Acid', 'Ash', 'Alkalinity_of_Ash', 'Magnesium',
    'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins',
    'Color_Intensity', 'Hue', 'OD280/OD315', 'Proline'
]
wine_data = pd.read_csv('wine\wine.data', header=None, names=column_names)

# Split data into features and target
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000)
}

# Metrics storage
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    end_time = time.time()
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Store performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC-AUC (for multi-class classification)
    y_prob = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')
    
    # Store results
    results[model_name] = {
        "Accuracy": accuracy,
        "Confusion Matrix": conf_matrix,
        "Classification Report": classification_rep,
        "Training Time": end_time - start_time,
        "ROC-AUC": auc_score
    }
    print(f"\n{model_name} - Completed Training")

# Print results summary
for model_name, metrics in results.items():
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Training Time: {metrics['Training Time']:.4f} seconds")
    print(f"ROC-AUC Score: {metrics['ROC-AUC']:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(metrics['Classification Report']).T)

# Plot ROC curves for comparison
plt.figure(figsize=(10, 8))

for model_name, model in models.items():
    y_prob = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)  # Assumes binary for plotting
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results[model_name]['ROC-AUC']:.4f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.show()
