import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "AIMS/Digital_Twin/chunks/chunks.json"
INDEX_FILE = "AIMS/Digital_Twin/vector_db/feynman.index"

print("Loading chunks...")

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

texts = [
    chunk["text"]
    for chunk in chunks
]

print(f"Loaded {len(texts)} chunks")

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

print(
    f"Embedding shape: {embeddings.shape}"
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    INDEX_FILE
)

print(
    f"Saved index to {INDEX_FILE}"
)