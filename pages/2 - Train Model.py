import os
import streamlit as st
import pdfplumber

from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def extract_text_from_pdfs(pdf_folder):
    all_text = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                all_text.append({"file": file, "text": text})
    return all_text

def split_documents(text_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    documents = []
    for doc in text_data:
        chunks = splitter.create_documents([doc["text"]])
        for chunk in chunks:
            chunk.metadata = {"source": doc["file"]}
        documents.extend(chunks)
    return documents

# --- Streamlit UI ---
# Sidebar
st.sidebar.caption(':orange[This app was built on debate reports from the Parliament of Ghana. Credit: github.com/agbozo1]')

col1, _, col3 = st.columns([3, 1, 2])
with col1:
    st.subheader('🧠 Train Model on 🇬🇭 Parliamentary Briefs')
with col3:
    st.image('imgs/Ghana_Parliament_Emblem.png', width=150)


st.write("This will read all PDF files in the `proceedings/` folder, extract text, split into chunks, generate embeddings, and save them in a FAISS vector DB.")

if st.button("Start Training"):
    with st.spinner("🔍 Extracting text from PDFs..."):
        pdf_folder = "proceedings"
        if not os.path.exists(pdf_folder) or not os.listdir(pdf_folder):
            st.error("❌ No PDF files found in the 'proceedings/' folder.")
        else:
            pdf_texts = extract_text_from_pdfs(pdf_folder)
            st.success(f"✅ Extracted text from {len(pdf_texts)} PDF(s).")

            st.info("📚 Splitting text into chunks...")
            documents = split_documents(pdf_texts)
            st.success(f"✅ Created {len(documents)} text chunks.")

            st.info("🔗 Generating embeddings with Ollama (`all-minilm:latest`)...")
            try:
                embeddings = OllamaEmbeddings(model="all-minilm:latest")
                vector_db = FAISS.from_documents(documents, embeddings)
                vector_db.save_local("parliament_faiss_db_allminilm")
                st.success("🎉 Model trained and vector DB saved as `parliament_faiss_db_allminilm/`!")
            except Exception as e:
                st.error(f"❌ Failed to generate embeddings: {e}")
