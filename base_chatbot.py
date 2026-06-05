import json
import faiss
import numpy as np
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import os

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

index = faiss.read_index(
    "AIMS/Digital_Twin/vector_db/feynman.index"
)

with open(
    "AIMS/Digital_Twin/chunks/chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

embedding_model = (
    SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
)

while True:

    question = input(
        "\nYou: "
    )

    if question.lower() == "exit":
        break

    query_embedding = (
        embedding_model.encode(
            [question]
        )
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = (
        index.search(
            query_embedding,
            k=5
        )
    )

    context = ""

    for idx in indices[0]:

        context += (
            chunks[idx]["text"]
            + "\n\n"
        )

    prompt = f"""
Use the context below to answer.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(
        prompt
    )

    print(
        "\nAssistant:"
    )

    print(
        response.text
    )