
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Testing a single embedding...")

vector = embeddings.embed_query(
    "Microsoft is a technology company."
)

print("Success!")
print("Vector dimensions:", len(vector))

print("\nTesting multiple embeddings...")

vectors = embeddings.embed_documents([
    "Microsoft develops Windows.",
    "Nvidia develops GPUs.",
    "Ollama runs language models locally."
])

print("Batch embedding successful!")
print("Number of vectors:", len(vectors))
