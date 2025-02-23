import pandas as pd
import umap
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

# Load data https://huggingface.co/datasets/MaartenGr/arxiv_nlp
print("Carregando o dataset....")
dataset = load_dataset("maartengr/arxiv_nlp")["train"][:5000]
print("Tamanho do dataset:", len(dataset))

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

##Common pipeline
# 1.Convert the input documents to embeddings with an embedding model.
##1.1 Select the embedding model
##  https://huggingface.co/spaces/mteb/leaderboard
print("Convertendo para embeddings....")
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
print("Embedding shape:", embeddings.shape)

# 2.Reduce the dimensionality of embeddings with a dimensionality reduction model.
# UMAP - Uniform Manifold Approximation and Projection - redução de dimensionalidade
# pip install umap-learn
print("Reduzindo a dimensionalidade....")
from umap import UMAP
umap_model = UMAP(n_components=5,
                  min_dist=0.0,
                  metric="cosine",
                  random_state=42 # reproducible across sessions, but slower
                  )
reduced_embeddings = umap_model.fit_transform(embeddings)

# 3.Find groups of semantically similar documents with a cluster model
#pip install hdbscan
from hdbscan import HDBSCAN

# Centroid-based vs Density-based
#HBDSCAN(Hierarchical Density-Based SpatiaL Clustering of Applications with Noise)
# is a density-based model
print("Procurando grupos de similaridades...")
hdbscan_model = HDBSCAN(
    min_cluster_size=50, #clusters are defined by the min cluster size
    metric="euclidean",
    cluster_selection_method="eom"
).fit(reduced_embeddings)

clusters = hdbscan_model.labels_
print("How many clusters were created:", len(set(clusters)))

# Inspecting the clusters
import numpy as np

cluster = 0
for index in np.where(clusters == cluster)[0][:3]:
    print(abstracts[index][:300] + "...\n")

# Reduce the dimensional embeddings to two dimensions for easier visualization
print("Reduzindo a dimensionalidade para representar visualmente de forma mais facil... ")
reduced_embeddings = UMAP(
    n_components=2,
    min_dist=0.0,
    metric="cosine",
    random_state=42
).fit_transform(embeddings)

df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]
to_plot = df.loc[df.cluster != "-1", :]
outliers = df.loc[df.cluster == "-1", :]

#Plotting
#A versão 3.10 do matplotlib tem incompatibilidade com o pycharm
#pip install matplotlib==3.9.0
import matplotlib.pyplot as plt

plt.scatter(outliers.x, outliers.y, alpha=0.05, s=2, c="grey")
plt.scatter(
    to_plot.x,
    to_plot.y,
    c=to_plot.cluster.astype(int),
    alpha=0.6,
    s=2,
    cmap="tab20b"
)
plt.axis("off")

plt.show()
