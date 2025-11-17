import os
import re
import time
import json  # For loading JSON Lines
from typing import List
from tqdm import tqdm  # For progress bar
import pickle

# =============================================
# 🔐 Hugging Face Authentication (Local Cache)
# =============================================
import os

# ⚠️ Replace this with your actual Hugging Face token if needed
HF_TOKEN = "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"

# --- Define a local cache path ---
LOCAL_CACHE_PATH = os.path.expanduser("~/.cache/huggingface")
print(f"[+] Setting Hugging Face cache directory to: {LOCAL_CACHE_PATH}")

os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = LOCAL_CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = LOCAL_CACHE_PATH

# Optional: Try to log in
try:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("[+] Hugging Face login successful (or token already cached).")
except ImportError:
    print("[+] huggingface_hub not found, skipping login. Embeddings may load from cache.")
except Exception as e:
    print(f"[!] Hugging Face login failed: {e}")

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


# --- 1. CONFIGURATION ---

# <--- Pointing to the multicare JSONL file ---
SOURCE_JSONL_FILE = "cases_json/chest_multicare_cases.jsonl" 

# <--- Output paths for this dataset ---
PERSIST_DIRECTORY_TEXT = "./chroma_db_medical_multicare"
DOCSTORE_PKL_FILE = "./docstore_medical_multicare.pkl"
CLEANED_TEXT_OUTPUT_FILE = "PROCESSED_MEDICAL_MULTICARE.txt"

# --- Embedding Model (Fast, Domain-Specific) ---
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# --- Chunking Strategy (from your old script) ---
PARENT_CHUNK_SIZE = 2048
CHILD_CHUNK_SIZE = 256


# --- 2. CLEANING & PARSING FUNCTIONS (MODIFIED) ---
def load_and_process_jsonl(jsonl_path: str) -> List[Document]:
    """
    Loads the multicare JSONL file, extracts 'case_text' as content,
    and other keys as metadata.
    """
    print(f"  > Loading & processing {jsonl_path}...")
    all_docs = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing multicare records"):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    
                    # Use 'case_text' as the main text content
                    case_content = record.get("case_text") 
                    
                    if not case_content or len(case_content) < 100:
                        continue

                    # Preprocessing: lowercase and normalize whitespace
                    clean_content = re.sub(r"\s+", " ", case_content).strip().lower()

                    # Extract metadata from the multicare keys
                    metadata = {
                        "case_id": record.get("case_id"),
                        "article_id": record.get("article_id"), 
                        "age": record.get("age"),
                        "gender": record.get("gender"),
                        "source_file": os.path.basename(jsonl_path)
                    }
                    
                    all_docs.append(
                        Document(
                            page_content=clean_content,
                            metadata=metadata
                        )
                    )
                except json.JSONDecodeError:
                    print(f"  [!] Warning: Skipping malformed JSON line.")
    
    except FileNotFoundError:
        print(f"  [!] Error: File not found {jsonl_path}")
        return []
    except Exception as e:
        print(f"  [!] Error reading file: {e}")
        return []

    return all_docs

# --- 3. MAIN INGESTION SCRIPT ---

if __name__ == "__main__":

    print("--- Starting RAG Pipeline B: Medical Multicare JSONL Ingestion ---")

    # --- Step 1: Load and Process All JSONL Records ---
    if not os.path.isfile(SOURCE_JSONL_FILE):
        print(f"Error: Source file not found: {SOURCE_JSONL_FILE}")
        exit()

    print(f"--- Processing file: {SOURCE_JSONL_FILE} ---")
    all_clean_docs = []
    try:
        docs_from_file = load_and_process_jsonl(SOURCE_JSONL_FILE)
        
        if docs_from_file:
            all_clean_docs.extend(docs_from_file)
            print(f"  > Loaded and processed {len(all_clean_docs)} records.")
        else:
            print(f"  > No clean documents extracted from {SOURCE_JSONL_FILE}.")

    except Exception as e:
        print(f"  > [!!!] FAILED to process {SOURCE_JSONL_FILE}: {e}")
        exit()

    if not all_clean_docs:
        print("No documents were processed. Exiting.")
        exit()

    print(f"\nTotal documents to be indexed: {len(all_clean_docs)}")
    
    # --- Step 2: Save Cleaned Text to PROCESSED.txt ---
    print(f"\n--- Step 2: Saving processed text to {CLEANED_TEXT_OUTPUT_FILE} ---")
    try:
        with open(CLEANED_TEXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for i, doc in enumerate(all_clean_docs):
                f.write(f"--- CASE_ID: {doc.metadata.get('case_id', 'unknown')}, ARTICLE_ID: {doc.metadata.get('article_id', 'unknown')} ---\n")
                f.write(f"--- AGE: {doc.metadata.get('age', 'N/A')}, GENDER: {doc.metadata.get('gender', 'N/A')} ---\n")
                f.write(doc.page_content)
                if i < len(all_clean_docs) - 1:
                    f.write("\n\n" + "="*80 + "\n\n")
        print(f"Successfully saved all processed text to {CLEANED_TEXT_OUTPUT_FILE}.")
    except Exception as e:
        print(f"Error saving processed text: {e}")

    # --- Step 3: Initialize LangChain Components ---
    print("\n--- Step 3: Initializing LangChain Components ---")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    
    # ⚡️ OPTIMIZATION: This is the batch size for EACH GPU
    EMBED_BATCH_SIZE = 512     
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # 🔴 THIS IS THE FIX: Enable multi-processing to use all GPUs
        multi_process=True, 
        # We remove model_kwargs={'device': 'cuda'} as it's not needed.
        # multi_process handles device allocation.
        encode_kwargs={'batch_size': EMBED_BATCH_SIZE} 
    )

    print(f"Initializing persistent vector store at: {PERSIST_DIRECTORY_TEXT}...")
    vectorstore = Chroma(
        collection_name="medical_multicare_text", 
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY_TEXT
    )

    print("Initializing in-memory docstore...")
    docstore = InMemoryStore()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE)

    print("Initializing ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # --- Step 4: Run the Embedding & Indexing (with Batching) ---
    
    # ⚡️ FIX: Reduced batch size to stay under Chroma's 5461 limit
    DOC_BATCH_SIZE = 32 
    
    print("\n--- Step 4: Starting Embedding and Indexing ---")

    def batch_list(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    start_time = time.perf_counter()
    print(f"Adding {len(all_clean_docs)} documents in batches of {DOC_BATCH_SIZE}...")
    
    for doc_batch in tqdm(batch_list(all_clean_docs, DOC_BATCH_SIZE), total=(len(all_clean_docs) // DOC_BATCH_SIZE) + 1):
        try:
            retriever.add_documents(doc_batch, ids=None)
        except Exception as e:
            print(f"[!] Error adding batch to retriever: {e}")
            print("Skipping this batch...")

    end_time = time.perf_counter()

    # --- Step 5: Manually Save the Docstore ---
    print("\n--- Step 5: Saving Persistent Docstore (Parent Chunks) ---")
    try:
        with open(DOCSTORE_PKL_FILE, "wb") as f:
            pickle.dump(docstore.store, f)
        print(f"Successfully saved docstore (parent chunks) to: {DOCSTORE_PKK_FILE}")
    except Exception as e:
        print(f"[!] Error saving docstore with pickle: {e}")

    print("\n--- Ingestion Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Persistent vector store (child chunks) saved to: {PERSIST_DIRECTORY_TEXT}")
    print(f"Persistent docstore (parent chunks) saved to: {DOCSTORE_PKL_FILE}")
