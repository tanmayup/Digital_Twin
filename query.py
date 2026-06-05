import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

INDEX_FILE = "AIMS/Digital_Twin/vector_db/feynman.index"
CHUNKS_FILE = "AIMS/Digital_Twin/chunks/chunks.json"

print("Loading index...")
index = faiss.read_index(INDEX_FILE)

print("Loading chunks...")
with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)

print("Loading embedding model...")
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

while True:

    query = input("\nQuestion: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        k=5
    )

    print("\nTop Results:\n")

    for rank, idx in enumerate(indices[0], start=1):

        chunk = chunks[idx]

        print(
            f"\nResult {rank}"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            chunk["text"][:500]
        )

        print(
            "\n" + "="*60
        )