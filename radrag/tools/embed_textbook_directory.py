import os
import re
import time
import glob  # <--- NEW IMPORT: For searching for files
from typing import List
import fitz  # For PyMuPDF
from tqdm import tqdm  # For progress bar
import pickle

# =============================================
# 🔐 Hugging Face Authentication (Local Cache)
# =============================================
import os

# ⚠️ Replace this with your actual Hugging Face token
HF_TOKEN = "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"

# --- Define a local cache path ---
LOCAL_CACHE_PATH = "/ssd_scratch/cvit/saket/.hf_cache"
print(f"[+] Setting Hugging Face cache directory to: {LOCAL_CACHE_PATH}")

os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = LOCAL_CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = LOCAL_CACHE_PATH
print("[+] Hugging Face authentication initialized with local cache.")

from huggingface_hub import login
try:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("[+] Hugging Face login successful.")
except Exception as e:
    print(f"[!] Hugging Face login failed: {e}")

# --- LANGCHAIN IMPORTS ---
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---

# <--- MODIFIED: Changed from a single file to a directory path ---
SOURCE_PDF_DIRECTORY = "/ssd_scratch/cvit/saket/radiology_textbooks"

# --- Output Paths (Unchanged) ---
PERSIST_DIRECTORY_TEXT = "/ssd_scratch/cvit/saket/textbook_rag/chroma_db_text"
DOCSTORE_PKL_FILE = "/ssd_scratch/cvit/saket/textbook_rag/docstore.pkl"
CLEANED_TEXT_OUTPUT_FILE = "/ssd_scratch/cvit/saket/textbook_rag/PROCESSED_ALL.txt" # Renamed to reflect all files

# --- Embedding Model (Unchanged) ---
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# --- Chunking Strategy (Unchanged) ---
PARENT_CHUNK_SIZE = 2048
CHILD_CHUNK_SIZE = 256

# --- 2. CLEANING & PARSING FUNCTIONS (Unchanged) ---
# This function is perfect as-is. We just call it multiple times.

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean_pdf(pdf_path: str) -> List[Document]:
    """
    Loads a SINGLE PDF file using PyMuPDF and returns a list of Document objects.
    This function remains unchanged.
    """
    print(f"  > Loading & extracting {pdf_path} with PyMuPDF (fitz)...")
    clean_docs = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [!] Error opening {pdf_path} with fitz: {e}")
        return []

    # print(f"  > Document has {doc.page_count} pages.") # Too verbose for a loop
    for i, page in enumerate(doc):
        page_text = page.get_text()
        clean_content = normalize_text(page_text)
        if len(clean_content) < 50:
            continue
        clean_docs.append(
            Document(
                page_content=clean_content,
                # The metadata now correctly stores which file this page came from
                metadata={"source": pdf_path, "page_number": i + 1}
            )
        )

    doc.close()

    # We'll print the summary from the main loop
    # print(f"  > Extracted {len(clean_docs)} clean text pages.")
    return clean_docs

# --- 3. MAIN INGESTION SCRIPT ---

if __name__ == "__main__":

    print("--- Starting RAG Pipeline A: Text Ingestion (Directory Mode) ---")

    # --- Step 1: Find and Process All PDFs in Directory ---
    # <--- MODIFIED: This entire step is rewritten ---

    if not os.path.isdir(SOURCE_PDF_DIRECTORY):
        print(f"Error: Source directory not found: {SOURCE_PDF_DIRECTORY}")
        print("Please update the SOURCE_PDF_DIRECTORY variable.")
        exit()

    # Use glob to find all PDF files recursively (in subfolders)
    pdf_search_path = os.path.join(SOURCE_PDF_DIRECTORY, "**", "*.pdf")
    all_pdf_files = glob.glob(pdf_search_path, recursive=True)

    if not all_pdf_files:
        print(f"No PDF files found in {SOURCE_PDF_DIRECTORY} or its subdirectories.")
        exit()

    print(f"Found {len(all_pdf_files)} PDF files to process.")

    all_clean_docs = [] # This will store docs from ALL files

    for pdf_path in all_pdf_files:
        # Use os.path.basename to get just the filename for cleaner logging
        print(f"\n--- Processing file: {os.path.basename(pdf_path)} ---")
        try:
            # Load all pages from the current PDF
            docs_from_file = load_and_clean_pdf(pdf_path)

            if docs_from_file:
                # Add the list of pages from this file to the master list
                all_clean_docs.extend(docs_from_file)
                print(f"  > Added {len(docs_from_file)} clean pages from this file.")
            else:
                print(f"  > No clean text extracted from {pdf_path}.")

        except Exception as e:
            print(f"  > [!!!] FAILED to process {pdf_path}: {e}")
            print("  > Skipping this file and continuing with the next one.")

    if not all_clean_docs:
        print("No clean text was extracted from *any* document. Exiting.")
        exit()

    print(f"\nTotal clean pages (from all files) to be indexed: {len(all_clean_docs)}")
    # --- End of Modified Step 1 ---

    # --- Step 2: Save Cleaned Text to PROCESSED.txt ---
    # (No change needed, it just works on the aggregated list)
    print(f"\n--- Step 2: Saving cleaned text to {CLEANED_TEXT_OUTPUT_FILE} ---")
    try:
        with open(CLEANED_TEXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for i, doc in enumerate(all_clean_docs):
                f.write(f"--- SOURCE: {doc.metadata.get('source', 'unknown')}, PAGE: {doc.metadata.get('page_number', 'unknown')} ---\n")
                f.write(doc.page_content)
                if i < len(all_clean_docs) - 1:
                    f.write("\n\n" + "="*80 + "\n\n")
        print(f"Successfully saved all clean text to {CLEANED_TEXT_OUTPUT_FILE}.")
    except Exception as e:
        print(f"Error saving processed text: {e}")

    # --- Step 3: Initialize LangChain Components ---
    # (No change needed)
    print("\n--- Step 3: Initializing LangChain Components ---")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )
    print(f"Initializing persistent vector store at: {PERSIST_DIRECTORY_TEXT}...")
    vectorstore = Chroma(
        collection_name="text_corpus",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY_TEXT
    )

    print("Initializing in-memory docstore...")
    docstore = InMemoryStore()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE,chunk_overlap=25)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # --- Step 4: Run the Embedding & Indexing (with Batching) ---
    # (No change needed, it just works on the aggregated list)
    print("\n--- Step 4: Starting Embedding and Indexing ---")

    PAGE_BATCH_SIZE = 32
    def batch_list(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    start_time = time.perf_counter()
    print(f"Adding {len(all_clean_docs)} pages in batches of {PAGE_BATCH_SIZE}...")
    for doc_batch in tqdm(batch_list(all_clean_docs, PAGE_BATCH_SIZE), total=len(all_clean_docs) // PAGE_BATCH_SIZE + 1):
        retriever.add_documents(doc_batch, ids=None)
    end_time = time.perf_counter()

    # --- Step 5: Manually Save the Docstore ---
    # (No change needed)
    print("\n--- Step 5: Saving Persistent Docstore (Library) ---")
    try:
        with open(DOCSTORE_PKL_FILE, "wb") as f:
            pickle.dump(docstore.store, f)
        print(f"Successfully saved docstore (parent chunks) to: {DOCSTORE_PKL_FILE}")
    except Exception as e:
        print(f"[!] Error saving docstore with pickle: {e}")

    print("\n--- Ingestion Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Persistent vector store saved to: {PERSIST_DIRECTORY_TEXT}")
    print(f"Persistent docstore saved to: {DOCSTORE_PKL_FILE}")
