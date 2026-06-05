# Richard Feynman Digital Twin

A Retrieval-Augmented Generation (RAG) based digital twin of Richard Feynman that combines his books, lectures, interviews, and essays with Google's Gemini model to create a chatbot capable of answering questions in a style inspired by Feynman's way of thinking and teaching.

---

## Project Overview

Richard Feynman was known not only for his contributions to physics, but also for his remarkable ability to explain complex ideas in simple and intuitive ways. The goal of this project is to build an AI system that can draw knowledge from Feynman's own writings and lectures while responding in a manner that reflects his curiosity, honesty, and teaching style.

Rather than relying entirely on a language model's built-in knowledge, the system retrieves relevant information from a curated collection of Feynman-related material before generating a response. This helps keep answers grounded in the ideas and explanations found in his work.

---

## Dataset Sources

The knowledge base currently includes content collected from several sources:

### Books

* Surely You're Joking, Mr. Feynman!
* What Do You Care What Other People Think?
* Finding Things Out

### Lectures

* The Feynman Lectures on Physics (Volumes I, II, and III)

### Interviews and Essays

* Cargo Cult Science
* There's Plenty of Room at the Bottom
* The Value of Science

### Quotes

* A curated collection of Richard Feynman quotes and remarks

---

## How It Works

The project follows a Retrieval-Augmented Generation (RAG) workflow.

### 1. Data Collection

Relevant books, lectures, interviews, essays, and quotes are gathered and converted into machine-readable text.

### 2. Text Processing

The collected material is cleaned and prepared for further processing.

### 3. Chunking

Large documents are divided into smaller overlapping chunks so that specific pieces of information can be retrieved efficiently.

### 4. Embedding Generation

Each text chunk is converted into a numerical vector representation using the Sentence Transformers model:

```text id="cfhnl9"
all-MiniLM-L6-v2
```

### 5. Vector Database

The generated embeddings are stored in a FAISS vector database, allowing fast similarity-based searches.

### 6. Retrieval

When a user asks a question:

1. The question is converted into an embedding.
2. FAISS retrieves the most relevant chunks from the knowledge base.
3. The retrieved information is assembled into a context block.

### 7. Response Generation

The retrieved context is combined with a custom Richard Feynman persona prompt and sent to Gemini, which generates the final answer.

---

## Technologies Used

### Programming Language

* Python

### Embedding Model

* Sentence Transformers
* all-MiniLM-L6-v2

### Vector Search

* FAISS

### Large Language Model

* Google Gemini 2.5 Flash

### Supporting Libraries

* NumPy
* JSON
* python-dotenv

---

## Current Features

* Automated document ingestion pipeline
* Text chunking and preprocessing
* Semantic embeddings using Sentence Transformers
* FAISS-based vector search
* Retrieval-Augmented Generation (RAG)
* Richard Feynman-inspired persona prompting
* Context-aware response generation
* Terminal-based chatbot interface

---

## Example Workflow

```text id="ujebao"
User Question
        ↓
Query Embedding
        ↓
FAISS Similarity Search
        ↓
Relevant Feynman Chunks
        ↓
Persona + Context Prompt
        ↓
Gemini
        ↓
Response
```

---

## Future Improvements

Some features planned for future development include:

* Interactive web interface
* Conversation memory
* Long-term user preferences and memory
* Source attribution and citations
* Multi-turn contextual conversations
* Improved persona consistency
* Response evaluation and benchmarking

---

## Author

This project was developed as part of a Digital Twin summer project with the objective of creating an AI representation of Richard Feynman's knowledge, communication style, and approach to teaching and learning.
