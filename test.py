import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import networkx as nx
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

### MODEL 1: RANDOM FOREST ###
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Feature importance from Random Forest
rf_importances = rf_model.feature_importances_

### MODEL 2: LOGISTIC REGRESSION ###
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

# Feature coefficients (importance) from Logistic Regression
log_importances = np.abs(log_model.coef_).mean(axis=0)

### MODEL 3: GRADIENT BOOSTING CLASSIFIER ###
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.2f}")

# Feature importance from Gradient Boosting
gb_importances = gb_model.feature_importances_

# Average importance across models
avg_importance = (rf_importances + log_importances + gb_importances) / 3
features = X.columns

### BUILD A FEATURE CORRELATION NETWORK ###
# Build a correlation matrix
correlation_matrix = pd.DataFrame(X_scaled, columns=features).corr()

# Create a network graph
G = nx.Graph()

# Add feature nodes with average importance as node size
for feature, importance in zip(features, avg_importance):
    G.add_node(feature, weight=importance)

# Add edges based on significant correlations
for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        if i < j:  # Avoid duplicate edges
            weight = correlation_matrix.loc[feature1, feature2]
            if abs(weight) > 0.5:  # Threshold for significant correlation
                G.add_edge(feature1, feature2, weight=weight)

# Visualize the feature network
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Consistent positioning for better visualization

# Node size scaled by average importance
node_sizes = [G.nodes[node]['weight'] * 1000 for node in G.nodes]
edge_widths = [abs(G[u][v]['weight']) * 5 for u, v in G.edges]

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# Add edge labels for correlation values
edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Simplified Feature Network with Average Feature Importance")
plt.show()

### PRINT MODEL PERFORMANCE ###
print("\nModel Performance Summary:")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")
print(f"Gradient Boosting Accuracy: {gb_accuracy:.2f}")
