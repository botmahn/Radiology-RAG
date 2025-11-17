# unified_radiology_retriever.py

import os
import torch
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel

# =====================================================
# 1️⃣ CRITICAL: ENVIRONMENT & CACHE CONFIGURATION
# =====================================================
# Ensures all Hugging Face assets go to your specified scratch directory.
HF_CACHE_DIR = "/ssd_scratch/cvit/saket/hf_cache"
HF_TOKEN = "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"

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
os.makedirs(HF_CACHE_DIR, exist_ok=True)


class UnifiedRadiologyRetriever:
    """
    A retriever based on the logic from unified_retrieval.py.
    It uses BioBERT to embed a text query and retrieves similar entries
    from a ChromaDB collection containing radiology text reports.
    """

    def __init__(self, chroma_dir: str, collection_name: str, images_root: str):
        """
        Initializes the retriever by loading the model and connecting to the database.

        Args:
            chroma_dir (str): Path to the ChromaDB persistent directory.
            collection_name (str): The name of the collection within the database.
            images_root (str): The root directory where source images are stored.
        """
        print("[Retriever] Initializing UnifiedRadiologyRetriever...")
        self.device = self._choose_device()
        self.images_root = images_root

        print("[Retriever] Loading BioBERT model...")
        self.tokenizer, self.model = self._load_biobert()

        print(f"[Retriever] Connecting to ChromaDB at '{chroma_dir}'...")
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(collection_name)
        print("[Retriever] Initialization complete.")

    def _choose_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_biobert(self, model_name: str = "dmis-lab/biobert-v1.1"):
        """Loads the BioBERT tokenizer and model, ensuring they use the cache."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=HF_CACHE_DIR
        )
        model = AutoModel.from_pretrained(
            model_name, cache_dir=HF_CACHE_DIR
        ).to(self.device).eval()
        return tokenizer, model

    def _embed_text(self, text: str) -> List[float]:
        """Embeds a single string of text using the loaded BioBERT model."""
        tok = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.model(**tok)
            cls_embedding = out.last_hidden_state[:, 0, :]  # [CLS] token
            normalized_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=-1)
            return normalized_embedding[0].detach().cpu().float().numpy().tolist()

    def _resolve_image_path(self, image_rel_path: str) -> str:
        """Constructs an absolute path to an image file."""
        if not image_rel_path:
            return ""
        # Handles cases where the path might already be absolute
        if os.path.isabs(image_rel_path):
            return image_rel_path
        return str(Path(self.images_root) / Path(image_rel_path).name)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant documents for a given text query.

        Args:
            query (str): The text to search for.
            k (int): The number of documents to retrieve.

        Returns:
            A list of dictionaries, where each dictionary represents a retrieved case.
        """
        print(f"[Retriever] Embedding query: '{query[:50]}...'")
        query_vector = self._embed_text(query)

        print(f"[Retriever] Querying collection for top {k} results...")
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )

        # Format the results into a clean list of dictionaries
        formatted_results = []
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            text_content = docs[i] if i < len(docs) else ""
            distance = dists[i] if i < len(dists) else -1.0
            image_path_rel = meta.get("image_path", "")
            
            formatted_results.append({
                "id": doc_id,
                "text": text_content,
                "image_path": self._resolve_image_path(image_path_rel),
                "distance": distance,
                "metadata": meta, # Include original metadata for flexibility
            })
            
        return formatted_results

# =====================================================
# Standalone Test Block
# =====================================================
if __name__ == "__main__":
    print("Running standalone retriever test...")
    
    # --- IMPORTANT: CONFIGURE THESE PATHS FOR YOUR SETUP ---
    CHROMA_DB_PATH = "/ssd_scratch/cvit/saket/rexgradient/texts"
    IMAGES_ROOT_PATH = "/ssd_scratch/cvit/saket/rexgradient_448xx_resized_images"
    COLLECTION = "text"

    try:
        retriever = UnifiedRadiologyRetriever(
            chroma_dir=CHROMA_DB_PATH,
            collection_name=COLLECTION,
            images_root=IMAGES_ROOT_PATH,
        )

        test_query = "patient with symptoms of pneumonia"
        retrieved_cases = retriever.retrieve(test_query, k=3)

        print(f"\n✅ Successfully retrieved {len(retrieved_cases)} cases for query: '{test_query}'")
        for i, case in enumerate(retrieved_cases, 1):
            print(f"\n--- Result {i} ---")
            print(f"  ID: {case['id']}")
            print(f"  Distance: {case['distance']:.4f}")
            print(f"  Image Path: {case['image_path']}")
            print(f"  Text: {case['text'][:200]}...")

    except Exception as e:
        print(f"\n❌ An error occurred during the test: {e}")
        print("Please ensure CHROMA_DB_PATH and IMAGES_ROOT_PATH are correct.")