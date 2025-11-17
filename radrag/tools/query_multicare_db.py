import os
import pickle
from typing import List, Dict, Any

# --- LANGCHAIN IMPORTS ---
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:
    print("Could not import from langchain_classic, trying langchain...")
    from langchain.retrievers import ParentDocumentRetriever

# =============================================
# 🔐 Hugging Face Authentication (From your script)
# =============================================
LOCAL_CACHE_PATH = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = LOCAL_CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = LOCAL_CACHE_PATH


class MedicalRAGRetriever:
    """
    A class to encapsulate the loading and querying of the medical RAG pipeline.

    This class loads the persistent Chroma vector store (child chunks) and
    the pickled InMemoryStore (parent chunks) to provide a single
    .search() method.
    """
    
    def __init__(
        self,
        vector_store_path: str = "./chroma_db_medical_multicare",
        docstore_path: str = "./docstore_medical_multicare.pkl",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        parent_chunk_size: int = 2048,
        child_chunk_size: int = 256,
        k: int = 3
    ):
        """
        Initializes the retriever by loading all necessary components.

        Args:
            vector_store_path: Path to the persistent Chroma DB directory.
            docstore_path: Path to the pickled InMemoryStore file.
            embedding_model: Name of the Hugging Face embedding model.
            parent_chunk_size: The chunk size used for parent documents.
            child_chunk_size: The chunk size used for child documents.
            k: The number of top results to return.
        """
        print(f"[+] Initializing MedicalRAGRetriever...")
        self.vector_store_path = vector_store_path
        self.docstore_path = docstore_path
        self.embedding_model_name = embedding_model
        self.k = k

        # 1. Load Embeddings
        self.embeddings = self._load_embeddings()

        # 2. Load Persistent Vector Store (Chroma)
        self.vectorstore = self._load_vector_store(self.embeddings)

        # 3. Load Persistent Docstore (Pickle)
        self.docstore = self._load_doc_store()
        
        # 4. Initialize Splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)

        # 5. Re-construct the ParentDocumentRetriever
        # We pass search_kwargs={"k": k} to control the number of
        # child documents retrieved, which in turn controls the number
        # of parent documents returned.
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            search_kwargs={"k": self.k}
        )
        print(f"[+] Retriever is ready to search (k={self.k}).")


    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Loads the Hugging Face embedding model."""
        print(f"  > Loading embedding model: {self.embedding_model_name}")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            multi_process=True,
            encode_kwargs={'batch_size': 32}
        )

    def _load_vector_store(self, embeddings: HuggingFaceEmbeddings) -> Chroma:
        """Loads the persistent Chroma vector store."""
        print(f"  > Loading vector store from: {self.vector_store_path}")
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(
                f"Chroma directory not found: {self.vector_store_path}. "
                "Please run the ingestion script first."
            )
        
        return Chroma(
            collection_name="medical_multicare_text",
            persist_directory=self.vector_store_path,
            embedding_function=embeddings
        )

    def _load_doc_store(self) -> InMemoryStore:
        """Loads the pickled InMemoryStore."""
        print(f"  > Loading docstore from: {self.docstore_path}")
        if not os.path.isfile(self.docstore_path):
            raise FileNotFoundError(
                f"Docstore file not found: {self.docstore_path}. "
                "Please run the ingestion script first."
            )
            
        docstore = InMemoryStore()
        try:
            with open(self.docstore_path, "rb") as f:
                docstore.store = pickle.load(f)
            print(f"  > Loaded {len(docstore.store)} parent documents.")
            return docstore
        except Exception as e:
            print(f"[!] Error loading docstore from pickle: {e}")
            raise

    def search(self, query: str) -> List[Document]:
        """
        Performs a search against the RAG pipeline.

        Args:
            query: The user's text query.

        Returns:
            A list of retrieved parent Documents (up to 'k' documents).
        """
        print(f"\n--- Searching for: '{query}' ---")
        # .invoke() is the standard way to call a retriever in LangChain
        return self.retriever.invoke(query)

# --- This block allows you to run this file directly for testing ---
if __name__ == "__main__":
    
    print("--- Testing MedicalRAGRetriever Class ---")
    
    try:
        # 1. Initialize the class
        # This one line does all the loading and setup.
        rag_retriever = MedicalRAGRetriever(k=3)

        # 2. Define a query
        test_query = "What are the key imaging features of COVID-19 pneumonia on a chest CT?"

        # 3. Run the search
        results = rag_retriever.search(test_query)

        # 4. Print results
        print(f"\n--- Found {len(results)} top results ---")
        for i, doc in enumerate(results):
            print(f"\n--- Document {i+1} (Case ID: {doc.metadata.get('case_id', 'N/A')}) ---")
            print(f"Source: {doc.metadata.get('source_file')}")
            
            # Print the first 400 characters of the content for brevity
            print("\nContent (Snippet):")
            print(doc.page_content[:400] + "...")
            print("="*80)

    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not run test: {e}")
        print("Please ensure you have run the ingestion script and the files are in the correct location.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
