import os
from pypdf import PdfReader

DATA_DIR = "AIMS/Digital_Twin/data"
OUTPUT_DIR = "AIMS/Digital_Twin/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


for root, dirs, files in os.walk(DATA_DIR):

    for file in files:

        file_path = os.path.join(root, file)

        if file.endswith(".pdf"):

            print(f"Processing: {file}")

            text = extract_pdf_text(file_path)

            output_name = (
                os.path.splitext(file)[0]
                + ".txt"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_name
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(text)

            print(
                f"Saved: {output_name}"
            )

        elif file.endswith(".txt"):

            print(f"Copying: {file}")

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as src:

                text = src.read()

            output_path = os.path.join(
                OUTPUT_DIR,
                file
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as dst:

                dst.write(text)

print("\nIngestion Complete.")