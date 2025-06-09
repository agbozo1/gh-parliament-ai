import streamlit as st
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
import datetime
import pandas as pd
import os

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


#QA chain with Ollama
qa_chain = RetrievalQA.from_llm(
    llm=OllamaLLM(model="llama3.1:latest"),
    retriever=retriever
)

# Function to persist queries
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




# Streamlit section
# Sidebar
st.sidebar.caption(':orange[This app was built on debate reports from the Parliament of Ghana. Credit: github.com/agbozo1]')

col1, _, col3 = st.columns([3, 1, 2])
with col1:
    st.subheader('Query 🇬🇭 Parliamentary Proceedings')
with col3:
    st.image('imgs/Ghana_Parliament_Emblem.png', width=150)


query = st.text_area(label='Prompt')
submit = st.button('Query')

if submit:
    if query.strip() == '':
        st.warning('Sorry! You cannot submit an empty query', icon="⚠️")
    else:
        with st.spinner("Gathering your results..."):
            response = qa_chain.invoke({"query": query})
        
        st.success("Response:")
        st.write(response)

        st.write("🗃️ Past Queries")
        datatable = persist_queries(q=query, response=response)
        st.dataframe(datatable)
