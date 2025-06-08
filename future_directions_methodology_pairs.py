from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings from cleaned combined text
embeddings = model.encode(df_merged['combined'].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ Cleaned and embedded {len(embeddings)} documents.")
print(f"Shape of embeddings: {embeddings.shape}")
print(f"FAISS index created with {index.ntotal} vectors.")
model = SentenceTransformer('all-mpnet-base-v2')

# prompt: based on the model,  can you generate a code for suggested visualisations please? 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- Visualizations ---

# 1. Distribution of Authors
# Count occurrences of each author. Authors list needs careful handling.
# We'll flatten the list of lists of authors first.
all_authors = [author for authors_list in df_merged['authors'] for author in authors_list]
author_counts = pd.Series(all_authors).value_counts()

# plt.figure(figsize=(12, 8))
# author_counts.head(20).plot(kind='bar') # Plot top 20 authors
# plt.title('Top 20 Most Frequent Authors')
# plt.xlabel('Author')
# plt.ylabel('Number of Papers')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # 2. Distribution of Paper Titles (Word Cloud - requires wordcloud library)
# !pip install wordcloud
# from wordcloud import WordCloud

text_titles = " ".join(df_merged['title'].fillna(''))

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_titles)

# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud of Paper Titles')
# plt.show()

# # 3. Embedding Visualization (using t-SNE for dimensionality reduction)
# # This can be computationally intensive for large datasets.
# # Let's sample or use a smaller subset if needed.
# # Using the 'all-mpnet-base-v2' embeddings from the last trained model

# if len(embeddings) > 500: # Limit for t-SNE visualization if dataset is too large
#     sample_indices = np.random.choice(len(embeddings), 500, replace=False)
#     embeddings_subset = embeddings[sample_indices]
#     titles_subset = df_merged['title'].iloc[sample_indices].tolist()
# else:
#     embeddings_subset = embeddings
#     titles_subset = df_merged['title'].tolist()


# print("Running t-SNE on embeddings (this may take a few minutes)...")
# tsne = TSNE(n_components=2, random_state=42, perplexity=4, n_iter=300) # Adjust perplexity and n_iter as needed
# embeddings_2d = tsne.fit_transform(embeddings_subset)


# plt.figure(figsize=(12, 10))
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
# plt.title('t-SNE Visualization of Paper Embeddings (Methodology & Future Work)')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')

# # Optionally add annotations for a few points (can make the plot crowded)
# # for i, title in enumerate(titles_subset[:50]): # Annotate first 50 points
# #     plt.annotate(title[:30] + '...', (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

# plt.show()

# # 4. Length Distribution of Methodology and Future Work Sections (after cleaning)
# df_merged['methodology_clean_len'] = df_merged['methodology_clean'].apply(lambda x: len(x.split()))
# df_merged['future_work_clean_len'] = df_merged['future_work_clean'].apply(lambda x: len(x.split()))

# plt.figure(figsize=(12, 6))
# sns.histplot(df_merged['methodology_clean_len'], bins=50, kde=True, color='skyblue', label='Methodology')
# sns.histplot(df_merged['future_work_clean_len'], bins=50, kde=True, color='salmon', label='Future Work')
# plt.title('Distribution of Section Lengths (Cleaned Text - Word Count)')
# plt.xlabel('Number of Words')
# plt.ylabel('Frequency')
# plt.legend()
# plt.xlim(0, 500) # Limit x-axis for better visualization if there are outliers
# plt.show()

# # 5. Example Snippets (Qualitative Check)
# print("\n--- Example Methodology Snippets ---")
# for i in range(min(5, len(df_merged))):
#     print(f"Paper: {df_merged.iloc[i]['title']}")
#     print(f"Snippet: {df_merged.iloc[i]['methodology_clean'][:500]}...\n")

# print("\n--- Example Future Work Snippets ---")
# for i in range(min(5, len(df_merged))):
#     print(f"Paper: {df_merged.iloc[i]['title']}")
#     print(f"Snippet: {df_merged.iloc[i]['future_work_clean'][:500]}...\n")



import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import faiss

# -----------------------------
# Assumptions:
# - df_merged: your DataFrame with columns ['title', 'authors', 'paper_id', 'methodology', 'future_work']
# - embeddings: np.ndarray of sentence embeddings for methodology+future_work
# - FAISS index already created
# -----------------------------

# Combine methodology and future work (if not already done)
if 'combined' not in df_merged.columns:
    df_merged['method_future_text'] = df_merged['methodology'].fillna('') + " " + df_merged['future_work'].fillna('')

# Optional: ensure correct embedding shape
assert embeddings.shape[0] == df_merged.shape[0], "Mismatch between embeddings and dataframe!"

# -----------------------------
# Step 1: Encode query and search nearest neighbors
# -----------------------------
query_text = "deep learning methods for formal theorem proving"
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode([query_text]).astype('float32')

# FAISS search
k = 5
distances, indices = index.search(query_embedding, k)
similar_indices = indices[0].tolist()

# -----------------------------
# Step 2: Flag similar papers in the dataframe
# -----------------------------
df_merged['is_similar'] = df_merged.index.isin(similar_indices)

# -----------------------------
# Step 3: Apply PCA for visualization
# -----------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca = pd.concat([df_pca, df_merged[['title', 'authors', 'paper_id', 'is_similar']].reset_index(drop=True)], axis=1)

# -----------------------------
# Step 4: Plot PCA results with highlights
# -----------------------------
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='is_similar', palette={True: 'red', False: 'blue'})
plt.title('PCA of Paper Embeddings (Top-K Similar Highlighted)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Top-K Similar?')
plt.grid(True)
plt.show()

# -----------------------------
# Step 5: Print Top-K Metadata
# -----------------------------
print("\n📌 Top-K Nearest Papers Based on Methodology/Future Work:\n")
for i in similar_indices:
    row = df_merged.iloc[i]
    print(f"🔹 Title   : {row['title']}")
    print(f"👥 Authors : {', '.join(row['authors']) if isinstance(row['authors'], list) else row['authors']}")
    print(f"🆔 Paper ID: {row.get('paper_id', 'N/A')}")
    print("-" * 60)
