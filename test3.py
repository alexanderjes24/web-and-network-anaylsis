import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from community import community_louvain

# Load dataset
column_names = [
    "Class", "Alcohol", "Malic_Acid", "Ash", "Alkalinity_of_Ash",
    "Magnesium", "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280/OD315", "Proline"
]
data = pd.read_csv("wine/wine.data", header=None, names=column_names)

# Generate synthetic textual descriptions
X = data.drop(columns=['Class'])
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Machine Learning Models for classification (optional, you can keep this for accuracy comparison)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "kNN": KNeighborsClassifier()
}

accuracies = []
for name, model in models.items():
<<<<<<< HEAD
    # Fit the model
=======
>>>>>>> parent of 2913aa2 (new changes)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracies.append((name, accuracy_score(y_test, y_pred)))

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


# ---- Network Analysis ----

# Step 1: Calculate pairwise similarity between wines using cosine similarity
similarity_matrix = cosine_similarity(X)

# Step 2: Create a Network Graph where each wine is a node and edges represent similarity
G = nx.Graph()

# Add nodes to the graph
for i in range(len(data)):
    G.add_node(i, label=f"Wine {i+1}")

# Add edges based on similarity threshold
threshold = 0.8  # Similarity threshold for creating an edge (can be adjusted)
for i in range(len(data)):
    for j in range(i+1, len(data)):
        if similarity_matrix[i][j] > threshold:
            G.add_edge(i, j, weight=similarity_matrix[i][j])

# Step 3: Visualize the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # Force-directed layout for visualization
nx.draw(G, pos, with_labels=True, node_size=300, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")
plt.title("Wine Similarity Network (Cosine Similarity)")
plt.show()

# Step 4: Network Analysis - Clustering or Centrality
# Optionally, compute centrality or clustering of the wines
centrality = nx.degree_centrality(G)
sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

# Display the top 5 wines with the highest centrality
print("Top 5 Wines by Degree Centrality:")
for i in range(5):
    print(f"Wines {sorted_centrality[i][0]+1}: Centrality {sorted_centrality[i][1]:.3f}")


partition = community_louvain.best_partition(G)

# Visualize the communities
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=300, cmap=plt.cm.Rainbow, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Wine Network Communities (Louvain)")
plt.show()