### THis code reads .csv file combines the strings of authors, abstract, title --> generate embeddings --> do cosine similarity --> plot UMAP

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt

# === Load data ===
df = pd.read_csv("./icml24_concat.csv")
df.fillna("", inplace=True)

# === Combine text fields ===
def combine_fields(row):
    return f"{row['title']}. {row['abstract']} Keywords: {', '.join(eval(row['tags']))} Authors: {', '.join(eval(row['name']))}"


df["text"] = df.apply(combine_fields, axis=1)
texts = df["text"].tolist()

# === Encode using sentence-transformers ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True)

# === Cluster into 2 groups ===
clusterer = AgglomerativeClustering(n_clusters=2, linkage='average', metric='cosine')
labels = clusterer.fit_predict(embeddings)
df["cluster"] = labels

# === UMAP for 2D visualization ===
reducer = umap.UMAP(metric='cosine', random_state=42)
embeds_2d = reducer.fit_transform(embeddings)

# === Plot with color by workshop ===
workshops = df["type"].unique()
color_map = {w: i for i, w in enumerate(workshops)}
colors = df["type"].map(color_map)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=colors, cmap='coolwarm', s=50, edgecolor='k')
handles = [plt.Line2D([0], [0], marker='o', color='w', label=w, 
                      markerfacecolor=plt.cm.coolwarm(i / (len(workshops)-1)), markersize=10)
           for i, w in enumerate(workshops)]
plt.legend(handles=handles, title="Workshop")
plt.title("UMAP projection of paper embeddings by Workshop")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.show()

# === Optional: How pure are the clusters w.r.t. workshop ===
df["workshop_id"] = df["type"].map(color_map)
purity = np.mean(df["cluster"] == df["workshop_id"])
print(f"\n🧪 Clustering purity (matching workshop): {purity:.2f} (1.0 = perfect separation)")
