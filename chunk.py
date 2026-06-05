from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

PROCESSED_DIR = "AIMS/Digital_Twin/processed"
OUTPUT_FILE = "AIMS/Digital_Twin/chunks/chunks.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_chunks = []

chunk_id = 0

for filename in os.listdir(PROCESSED_DIR):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(
        PROCESSED_DIR,
        filename
    )

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    chunks = splitter.split_text(text)

    for chunk in chunks:

        all_chunks.append(
            {
                "chunk_id": chunk_id,
                "source": filename,
                "text": chunk
            }
        )

        chunk_id += 1

print(
    f"Created {len(all_chunks)} chunks"
)

os.makedirs(
    "chunks",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Saved chunks.")