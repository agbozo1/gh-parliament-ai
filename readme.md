
# 🏛️ Ghana Parliamentary Debates QA App

This Streamlit application provides an end-to-end pipeline for downloading, processing, embedding, and querying parliamentary debate reports from the Parliament of Ghana.

---

## 📌 Overview

The application enables users to:

1. **Download Parliamentary PDFs** between a range of dates from the official Parliament of Ghana website.
2. **Extract & preprocess** text data from the PDFs.
3. **Train a vector database** using sentence embeddings.
4. **Query** the debate reports in natural language using a Retrieval-Augmented Generation (RAG) approach powered by LangChain and Ollama.

---

## 🗂️ Project Structure

```text
📁 gh_parliament_ai_app/
├── app.py                   # Main Streamlit app entry
├── pages/
│   ├── 1 - Download Briefs.py  # Page to download PDFs
│   └── 2 - Train Model.py      # Page to extract, split, embed and save vector DB
    └── 3 - Query Briefs.py     # Page to run queries on the RAG model
├── proceedings/            # Folder where downloaded PDFs are stored
├── parliament_faiss_db_allminlm   # Saved FAISS vector databases
```

---

## ⚙️ Features

### 📥 Docs Downloader (Page 1)

- Select a **start and end date**.
- Downloads all available parliamentary brief PDFs between those dates.
- Files are saved in the `proceedings/` folder.
- If a file already exists, it will be **overwritten**.

### 🧠 Train Model (Page 2)

- Reads PDFs from the `proceedings/` folder.
- Extracts and splits the text into manageable chunks.
- Generates **sentence-level embeddings** using `OllamaEmbeddings` (e.g., MiniLM or LLaMA3).
- Saves a local **FAISS vector store** for fast retrieval.

### ❓ Query Debate Reports (Page 3)

- Loads the trained vector store.
- Accepts user input in natural language.
- Retrieves relevant chunks using semantic search.
- Uses an LLM to answer questions based on retrieved context.

---

## 🛠️ Tech Stack

- **Streamlit** – Interactive web app framework.
- **LangChain** – RAG orchestration, embeddings, and vector store support.
- **Ollama** – For running LLMs locally like LLaMA3 and MiniLM.
- **FAISS** – Fast similarity search for embeddings.
- **pdfplumber** – Text extraction from PDFs.

---

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/agbozo1/gh-parliament-ai.git
   cd gh-parliament-ai
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   conda create -n streamlit python=3.10
   conda activate streamlit
   pip install -r requirements.txt
   ```

3. Make sure you have [Ollama](https://ollama.com) installed and running a model:

   ```bash
   ollama run llama3
   ```
   Also, make sure you have installed the following LLMs within Ollama:
    - **llama3.1:latest** (current embedding model used for this project)
    - **all-minilm:latest**

4. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 📚 Example Use Case

> **Query:** "What were the key economic reforms discussed in February 2024?"

The app will retrieve relevant passages from debate PDFs, and the LLM will generate a summarized answer based on context.

---

## 📎 Notes

- The app checks for the existence of files and **overwrites them** if already present.
- The PDF date format must match the Parliament of Ghana's naming convention (e.g., `1st January, 2024.pdf`).
- You can switch embedding models by changing the model name in `OllamaEmbeddings`.

---

## 🙌 Credits

Built using publicly available resources from the [Parliament of Ghana](https://www.parliament.gh) to promote transparency and civic engagement.

---
by: Ebenezer Agbozo, PhD.