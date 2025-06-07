from ingestion.load_docs import load_and_split_documents
from retrieval.faiss_handler import create_vectorstore

from langchain_community.chat_models import ChatOpenAI

def main():
    # Step 1: Load & split documents
    docs = load_and_split_documents("state_of_the_union.txt")

    # Step 2: Create vector store from docs
    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 3: Create RAG chain and run query
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    query = "What is this about?"
    result = qa_chain({"query": query})

    print("\nAnswer:\n", result["result"])
    print("\nSource documents:\n", result["source_documents"])

if __name__ == "__main__":
    main()
