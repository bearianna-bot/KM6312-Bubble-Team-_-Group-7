import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df_emb = pd.read_csv("eff_embeddings.csv")

features = df_emb[[c for c in df_emb.columns if c.startswith("eff_")]].values

tsne = TSNE(n_components=2, learning_rate='auto', perplexity=10)
X_2d = tsne.fit_transform(features)

df_emb["tsne_x"] = X_2d[:, 0]
df_emb["tsne_y"] = X_2d[:, 1]

plt.figure(figsize=(10,8))
scatter = plt.scatter(
    df_emb["tsne_x"],
    df_emb["tsne_y"],
    c=df_emb["posting_id"].astype('category').cat.codes,
    cmap="tab20",
    s=30
)
plt.colorbar(scatter)
plt.title("t-SNE Visualization (colored by posting_id)")
plt.show()

from PIL import Image
import imagehash
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import combinations

df = pd.read_csv("dataset_with_phash.csv")

sample = df.sample(5000, random_state=42)
phashes = sample["phash"].tolist()

def hamming(a, b):
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)

pairs = combinations(phashes, 2)
distances = [hamming(a,b) for a,b in pairs]

plt.figure(figsize=(8,5))
plt.hist(distances, bins=20)
plt.title("Histogram of pHash Hamming Distances (5000-sample)")
plt.xlabel("Hamming Distance")
plt.ylabel("Count")
plt.show()