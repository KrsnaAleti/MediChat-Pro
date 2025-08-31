# Code to create/store the index for FAISS and retreive the relevant documents

# langchain vectorstores documentation: https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

def create_faiss_index(texts):
    """
    Creates a FAISS index from a list of texts using HuggingFace embeddings.
    Args:
        texts (List[str]): A list of text strings to be indexed.
    Returns:
        FAISS: A FAISS vectorstore index.
    """
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Create the FAISS index
    faiss_index = FAISS.from_texts(texts, embeddings)
    
    return faiss_index

def retrive_relevant_docs(vectorstore: FAISS, query: str, k: int = 4):
    """
    Retrieves the most relevant documents from a FAISS vectorstore based on a query.

    Args:
        vectorstore (FAISS): The FAISS vectorstore to search within.
        query (str): The query string to find similar documents for.
        k (int, optional): The number of top relevant documents to retrieve. Defaults to 4.

    Returns:
        List[Document]: A list of Langchain Document objects representing the relevant documents.
    """

    return vectorstore.similarity_search(query, k=k)

    
