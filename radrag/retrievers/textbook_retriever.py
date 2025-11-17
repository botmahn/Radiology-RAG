# import os
# import pickle
# from typing import List

# # --- LANGCHAIN IMPORTS ---
# from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.stores import InMemoryStore
# from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # This class handles loading the pre-built RAG components and retrieving documents.
# class TextbookQARetriever:
#     def __init__(self, chroma_path: str, docstore_path: str, embedding_model_name: str):
#         """
#         Initializes the retriever by loading the Chroma vectorstore and the pickled docstore.
#         """
#         print("[TextbookRetriever] Initializing...")
#         if not os.path.isdir(chroma_path):
#             raise FileNotFoundError(f"Chroma DB path not found: {chroma_path}")
#         if not os.path.isfile(docstore_path):
#             raise FileNotFoundError(f"Docstore pickle file not found: {docstore_path}")

#         # 1. Load the embedding model
#         print(f"[TextbookRetriever] Loading embedding model: {embedding_model_name}")
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name=embedding_model_name,
#             model_kwargs={'device': 'cuda'}
#         )

#         # 2. Load the persistent Chroma vector store
#         print(f"[TextbookRetriever] Loading Chroma vectorstore from: {chroma_path}")
#         vectorstore = Chroma(
#             collection_name="text_corpus",
#             embedding_function=self.embeddings,
#             persist_directory=chroma_path
#         )

#         # 3. Load the docstore from the pickle file
#         print(f"[TextbookRetriever] Loading docstore from: {docstore_path}")
#         try:
#             with open(docstore_path, "rb") as f:
#                 loaded_store_dict = pickle.load(f)
            
#             self.docstore = InMemoryStore()
#             self.docstore.mset(list(loaded_store_dict.items())) # Re-populate the store
#             print("[TextbookRetriever] Docstore loaded successfully.")
#         except Exception as e:
#             raise IOError(f"Failed to load or parse the docstore pickle file: {e}")

#         # 4. Re-create the ParentDocumentRetriever with the loaded components
#         self.retriever = ParentDocumentRetriever(
#             vectorstore=vectorstore,
#             docstore=self.docstore,
#             child_splitter=RecursiveCharacterTextSplitter(chunk_size=256), # Dummy splitter
#         )
#         print("[TextbookRetriever] Initialization complete.")

#     def retrieve(self, query: str, k: int = 10) -> List[Document]:
#         """
#         Retrieves relevant documents from the textbook knowledge base.
#         """
#         print(f"[TextbookRetriever] Retrieving top {k} documents for query: '{query}'")
#         relevant_docs = self.retriever.invoke(
#             query, 
#             config={"configurable": {"search_kwargs": {"k": k}}}
#         )
#         print(f"[TextbookRetriever] Found {len(relevant_docs)} relevant documents.")
#         #print(relevant_docs)
#         return relevant_docs


import os
import pickle
from typing import List

# --- LANGCHAIN IMPORTS ---
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- NEW IMPORT FOR RERANKING ---
# We use this to load the reranking model
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# This class handles loading the pre-built RAG components and retrieving documents.
class TextbookQARetriever:
    def __init__(self, chroma_path: str, docstore_path: str, embedding_model_name: str):
        """
        Initializes the retriever by loading the Chroma vectorstore, the pickled docstore,
        the embedding model (for retrieval), AND the reranker model.
        
        The __init__ signature is kept identical to your original code.
        """
        print("[TextbookRetriever] Initializing...")
        if not os.path.isdir(chroma_path):
            raise FileNotFoundError(f"Chroma DB path not found: {chroma_path}")
        if not os.path.isfile(docstore_path):
            raise FileNotFoundError(f"Docstore pickle file not found: {docstore_path}")

        # 1. Load the embedding model (YOUR BIOMEDICAL MODEL)
        print(f"[TextbookRetriever] Loading embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'}
        )

        # 2. Load the persistent Chroma vector store
        print(f"[TextbookRetriever] Loading Chroma vectorstore from: {chroma_path}")
        vectorstore = Chroma(
            collection_name="text_corpus",
            embedding_function=self.embeddings, # <-- Uses your biomedical embeddings
            persist_directory=chroma_path
        )

        # 3. Load the docstore from the pickle file
        print(f"[TextbookRetriever] Loading docstore from: {docstore_path}")
        try:
            with open(docstore_path, "rb") as f:
                loaded_store_dict = pickle.load(f)
            
            self.docstore = InMemoryStore()
            self.docstore.mset(list(loaded_store_dict.items())) # Re-populate the store
            print("[TextbookRetriever] Docstore loaded successfully.")
        except Exception as e:
            raise IOError(f"Failed to load or parse the docstore pickle file: {e}")

        # 4. Re-create the ParentDocumentRetriever with the loaded components
        # This is our "base" retriever that uses your biomedical model
        self.base_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=self.docstore,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=256), # Dummy splitter
        )
        print("[TextbookRetriever] Base retriever created.")
        
        # --- 5. NEW: LOAD THE RERANKER MODEL ---
        # This is a separate model loaded for the reranking step.
        reranker_model_name = "BAAI/bge-reranker-base"
        print(f"[TextbookRetriever] Loading reranker model: {reranker_model_name}")
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name=reranker_model_name,
            model_kwargs={'device': 'cuda'}
        )
        
        print("[TextbookRetriever] Initialization complete.")

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieves relevant documents from the textbook knowledge base.
        
        This now performs a 2-stage process internally:
        1. Retrieve a "wide net" of documents (k * 5, or 25 minimum) using the base_retriever.
        2. Rerank these documents using the cross-encoder.
        3. Return the final top 'k' documents.
        """
        print(f"[TextbookRetriever] Reranking enabled. Final 'k' requested: {k}")

        # --- 1. RETRIEVE A "WIDE NET" OF DOCUMENTS ---
        initial_k = max(k * 5, 25) 
        print(f"[TextbookRetriever] Retrieving top {initial_k} docs for reranking...")
        
        retrieved_docs = self.base_retriever.invoke(
            query, 
            config={"configurable": {"search_kwargs": {"k": initial_k}}}
        )
        
        if not retrieved_docs:
            print("[TextbookRetriever] No documents found.")
            return []

        print(f"[TextbookRetriever] Found {len(retrieved_docs)} docs. Now reranking...")

        # --- 2. RERANK THE RETRIEVED DOCUMENTS ---
        
        # ***FIX 1:*** The .score() method needs a list of TUPLES (query, text)
        doc_texts = [doc.page_content for doc in retrieved_docs]
        query_doc_pairs = [(query, text) for text in doc_texts]

        # ***FIX 2:*** Use the .score() method provided by the class
        scores = self.reranker_model.score(query_doc_pairs)

        # --- 3. COMBINE DOCS, SORT, AND RETURN TOP 'k' ---
        doc_score_pairs = list(zip(retrieved_docs, scores))
        reranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in reranked_docs]
        
        print(f"[TextbookRetriever] Reranking complete. Returning top {k} documents.")
        return final_docs[:k]