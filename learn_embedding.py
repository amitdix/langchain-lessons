from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Example sentences
sentences = [
    "I love machine learning.",
    "Artificial intelligence is fascinating.",
    "Let's go to the beach!",
    "The ocean is beautiful today.",
    "I enjoy learning new languages.",
    "Studying deep learning is fun."
]

# 3. Generate embeddings
embeddings = model.encode(sentences)
print("embeddings",embeddings)
# 4. Reduce dimensions to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=2 if len(sentences) > 2 else 1)
embeddings_2d = tsne.fit_transform(embeddings)

# 5. Plot the embeddings
plt.figure(figsize=(8, 6))
for i, sentence in enumerate(sentences):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, sentence, fontsize=9)

plt.title("2D Visualization of Sentence Embeddings")
plt.grid(True)
plt.show()