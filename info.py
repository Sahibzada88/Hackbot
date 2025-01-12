import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

def pdf_database(question):
    file_path = "files/pdf/"
    pdf_files = [os.path.join(file_path,f) for f in os.listdir(file_path) if f.endswith('.pdf')]

    pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages.extend(loader.load())

    vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())
    documents = vector_store.similarity_search(question, 1)

    return " ".join([doc.page_content for doc in documents])




