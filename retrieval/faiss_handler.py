from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def create_vectorstore(docs):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore