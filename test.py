import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
column_names = [
    'Class', 'Alcohol', 'Malic_Acid', 'Ash', 'Alkalinity_of_Ash', 'Magnesium',
    'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins',
    'Color_Intensity', 'Hue', 'OD280/OD315', 'Proline'
]
wine_data = pd.read_csv('wine/wine.data', header=None, names=column_names)

# Split data into features and target
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_
features = X.columns

# Build a correlation matrix for features
correlation_matrix = pd.DataFrame(X_scaled, columns=features).corr()

# Create a network graph
G = nx.Graph()

# Add feature nodes and their importances as node weights (filter by importance > 0.05)
important_features = {
    feature: importance for feature, importance in zip(features, feature_importances) if importance > 0.05
}
for feature, importance in important_features.items():
    G.add_node(
        feature,
        weight=importance,
        description=f"Importance: {importance:.2f}"
    )

# Add edges based on significant correlations (filter by abs(correlation) > 0.5)
for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        if i < j and feature1 in important_features and feature2 in important_features:
            weight = correlation_matrix.loc[feature1, feature2]
            if abs(weight) > 0.5:  # Significant correlation threshold
                G.add_edge(
                    feature1, feature2,
                    weight=weight,
                    description=f"Correlation: {weight:.2f}"
                )

# Visualize the network
plt.figure(figsize=(14, 10))

# Use a circular layout for better spacing
pos = nx.circular_layout(G)

# Draw nodes with size proportional to feature importance
node_sizes = [G.nodes[node]['weight'] * 2000 for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)

# Draw edges with width proportional to correlation
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[abs(weight) * 3 for weight in weights], alpha=0.6)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# Add edge labels for correlation values
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)},
    font_size=8
)

# Add title and explanation
plt.title(
    "Simplified Feature Network for Wine Dataset\n"
    "Nodes: Important Features (Importance > 0.05)\n"
    "Edges: Significant Correlations (|Correlation| > 0.5)",
    fontsize=14
)
plt.show()
