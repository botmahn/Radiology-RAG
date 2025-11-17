import os
import textwrap
import pickle # For loading the docstore

# --- We use the 'langchain_community' imports that we know are in your environment ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================
# 🔐 Hugging Face Authentication (Local Cache)
# =============================================
import os

# --- Define a local cache path ---
LOCAL_CACHE_PATH = "/ssd_scratch/cvit/saket/.hf_cache" 
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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" # Your Token
os.environ["HF_HUB_TOKEN"] = "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" # Your Token
os.environ["HF_HOME"] = LOCAL_CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = LOCAL_CACHE_PATH

# =============================================
# 1. CONFIGURATION (Must match ingestion)
# =============================================
PERSIST_DIRECTORY_TEXT = "./chroma_db_text"
DOCSTORE_PKL_FILE = "./docstore.pkl" # <-- Path to our saved docstore
EMBEDDING_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT" # Must be the same as ingestion
PARENT_CHUNK_SIZE = 2048
CHILD_CHUNK_SIZE = 256

def initialize_retriever():
    """
    Initializes and returns a fully configured ParentDocumentRetriever.
    """
    print("--- Initializing Retriever Components ---")
    
    # 1. Load Embedding Model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )

    # 2. Load Persistent Vector Store (The "Search Index")
    print(f"Loading persistent vector store from: {PERSIST_DIRECTORY_TEXT}...")
    if not os.path.exists(PERSIST_DIRECTORY_TEXT):
        print(f"Error: Vector store not found at {PERSIST_DIRECTORY_TEXT}")
        print("Please run the 'process_texbook.py' script first.")
        return None
        
    vectorstore = Chroma(
        collection_name="text_corpus",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY_TEXT
    )

    # 3. Load Persistent Docstore (The "Library") from Pickle
    print(f"Loading persistent docstore from: {DOCSTORE_PKL_FILE}...")
    if not os.path.exists(DOCSTORE_PKL_FILE):
        print(f"Error: Docstore file not found at {DOCSTORE_PKL_FILE}")
        print("Please re-run the 'process_texbook.py' script to create it.")
        return None
        
    docstore = InMemoryStore()
    try:
        with open(DOCSTORE_PKL_FILE, "rb") as f:
            store_data = pickle.load(f)
            docstore.store = store_data
    except Exception as e:
        print(f"[!] Error loading docstore from pickle: {e}")
        return None

    # 4. Initialize Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE)

    # 5. Initialize the Retriever
    print("Initializing ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    print("--- 📚 Retriever is ready to query! ---")
    return retriever

def wrap_text(text, width=100):
    """Helper function to format long text for printing."""
    return '\n'.join(textwrap.wrap(text, width=width, replace_whitespace=False))

def main():
    retriever = initialize_retriever()
    if retriever is None:
        return

    try:
        while True:
            print("\n" + "="*80)
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            if not query:
                continue

            print(f"\n🔍 Searching for: '{query}'...")
            
            # --- **** THIS IS THE FIX **** ---
            retrieved_docs = retriever.invoke(query)
            # --- **** END OF FIX **** ---
            
            if not retrieved_docs:
                print("No relevant documents found.")
                continue

            print(f"\nFound {len(retrieved_docs)} relevant document chunks:")
            
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page_number', 'Unknown Page')
                
                print("\n---")
                print(f"**Result {i+1}** (Source: {os.path.basename(source)}, Page: {page})")
                print("---")
                print(wrap_text(doc.page_content))
                
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()