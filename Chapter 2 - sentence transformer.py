#pip install sentence-transformers
from sentence_transformers import SentenceTransformer

#Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Convert text to texto embeddings
vector = model.encode("Best movie ever!")

print("Shape:")
print(vector.shape)

print(vector)

