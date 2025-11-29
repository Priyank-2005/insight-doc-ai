# InsightDoc AI
### *Turn your static PDFs into an interactive conversation.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-4285F4)

---

## What is this?
Ever stared at a 50-page technical manual or a dense research paper and wished you could just **ask it questions**? 

**InsightDoc AI** is a RAG (Retrieval-Augmented Generation) application that lets you chat with your PDF documents. It doesn't just "guess" answers; it reads your specific files, finds the exact paragraph containing the answer, and uses Google's **Gemini 2.5** to summarize it for you.

**The "Secret Sauce":** unlike most tutorials that burn through API credits, this project runs the **Embedding layer locally (on your CPU)** using HuggingFace models. This means **zero cost** for processing documents and **faster** performance.

---

## Key Features
* **Multi-Document Support:** Upload multiple PDFs at once and query them all simultaneously.
* **Zero-Cost Embeddings:** Uses `all-MiniLM-L6-v2` (running locally) instead of paid OpenAI/Google embedding APIs.
* **Intelligent Memory:** The chat remembers your previous questions (Session State), so you can ask follow-ups.
* **Hybrid Architecture:** Combines local privacy (FAISS Vector DB) with cloud intelligence (Google Gemini).
* **Professional UI:** Clean Streamlit interface with status indicators, clear buttons, and error handling.

---

## Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/) (UI/UX)
* **LLM:** Google Gemini 2.5 Flash (via `google-generativeai`)
* **Orchestration:** [LangChain](https://www.langchain.com/) (Document loading & splitting)
* **Vector Database:** FAISS (Facebook AI Similarity Search) running locally
* **Embeddings:** HuggingFace `sentence-transformers` (Local CPU)

---

## How to Run Locally

Follow these steps to get InsightDoc AI running on your machine in 5 minutes.

### 1. Clone the Repo
git clone [https://github.com/Priyank-2005/insight-doc-ai.git](https://github.com/Priyank-2005/insight-doc-ai.git)
cd insight-doc-ai

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Set up your API Key
Create a file named .env in the root folder.
Add your key inside:
GOOGLE_API_KEY="AIzaSy...YOUR_KEY_HERE"

### 4. Run the App
streamlit run app.py

---

## How It Works (The "RAG" Pipeline)

- **Ingestion**: The app reads your PDFs using PyPDF2.
- **Chunking**: It splits the massive text into smaller, manageable chunks (10,000 characters) using RecursiveCharacterTextSplitter.
- **Embedding**: These chunks are converted into mathematical vectors using the HuggingFace model. This happens on your laptop!
- **Storage**: The vectors are stored in a local FAISS index (a searchable database).
- **Retrieval**: When you ask a question, the app searches the FAISS index for the most similar text chunks.
- **Generation**: It sends your question + the relevant text chunks to Google Gemini, which generates the final human-like answer.

---

Created by [Priyank Bohra] | Data Analyst & Python Developer

