import streamlit as st
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
import datetime
import pandas as pd
import os

# Workaround for OMP conflict (⚠️ use with caution)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize embeddings
embeddings = OllamaEmbeddings(model="all-minilm:latest")


vector_db = FAISS.load_local(
    folder_path='parliament_faiss_db_allminilm',
    embeddings=embeddings,
    index_name='index',
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever()


# Create QA chain with Ollama
qa_chain = RetrievalQA.from_llm(
    llm=OllamaLLM(model="llama3.1:latest"),
    retriever=retriever
)

# persist queries
def persist_queries(q, response):
    try:
        df = pd.read_csv('persisted_queries.csv')
    except FileNotFoundError:
        st.write('No Database exists. Creating DB')
        df = pd.DataFrame(columns=['timestamp', 'query', 'response'])
    
    new_entry = pd.DataFrame(
        [[datetime.datetime.now(), q, response]],
        columns=['timestamp', 'query', 'response']
    )
    
    df = pd.concat([new_entry, df], ignore_index=True)
    df.to_csv('persisted_queries.csv', index=False)
    return df

# Sidebar
st.sidebar.caption(':orange[This app was built on debate reports from the Parliament of Ghana. Credit: github.com/agbozo1]')

st.write("""
### 🏛️ Ghana Parliamentary Debates QA App
The application enables users to:
1. **Download Parliamentary PDFs** between a range of dates from the official Parliament of Ghana website.
        
2. **Extract & preprocess** text data from the PDFs.
         
3. **Train a vector database** using sentence embeddings.
         
4. **Query** the debate reports in natural language using a Retrieval-Augmented Generation (RAG) approach powered by LangChain and Ollama.

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

---""")

st.image('imgs/ghparliamentmodel-diagram.jpg')