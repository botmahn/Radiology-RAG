# medical_multicare_retriever.py
# pip install "langchain>=0.2" langchain-community chromadb langchain-text-splitters huggingface_hub
# Optional (for reranking): pip install langchain-ollama langsmith

from __future__ import annotations
import os
import pickle
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
try:
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain_classic.retrievers import ParentDocumentRetriever

import numpy as np

os.environ["HF_HOME"] = "/Users/namanmishra/Documents/Code/iiith_courses/lma/major_project/Radiology-RAG/pretrained_models"

# Compression / Reranking utilities
try:
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    from langchain_classic.retrievers import ContextualCompressionRetriever
    _HAS_COMPRESSION = True
except Exception:
    _HAS_COMPRESSION = False

# Ollama LLM (prefer Chat model; fallback to legacy LLM)
_OllamaChat = None
_OllamaLLM = None
try:
    # New-style
    from langchain_ollama import ChatOllama as _OllamaChat
except Exception:
    try:
        # Legacy
        from langchain_community.llms import Ollama as _OllamaLLM
    except Exception:
        pass

# LangSmith trace decorator (optional / no-op fallback)
try:
    from langsmith import traceable
except Exception:
    def traceable(**_kwargs):
        def _wrap(fn):
            return fn
        return _wrap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MedicalMulticareRetriever:
    """
    Build a ParentDocumentRetriever from your persisted Chroma store + pickled docstore,
    retrieve results, and (optionally) LLM-rerank them.

    JSON schema per result:
    {
      "case_id": str | None,
      "article_id": str | None,
      "age": int | str | None,
      "gender": str | None,
      "text": str
    }
    """

    def __init__(
        self,
        persist_dir: str,
        docstore_pkl: str,
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        collection_name: str = "medical_multicare_text",
        parent_chunk_size: int = 2048,
        child_chunk_size: int = 256,
        hf_token_env_keys: Optional[List[str]] = None,
        # Rerank defaults
        rerank_model: str = "llama3.2:3b",
        rerank_temperature: float = 0.0,
    ) -> None:
        self.persist_dir = persist_dir
        self.docstore_pkl = docstore_pkl
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.rerank_model = rerank_model
        self.rerank_temperature = rerank_temperature

        # Optional HF login (uses cached weights if public)
        token = None
        if hf_token_env_keys is None:
            hf_token_env_keys = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"]
        for k in hf_token_env_keys:
            if os.environ.get(k):
                token = os.environ.get(k)
                break
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
            except Exception:
                pass  # non-fatal

        # Embeddings (must match indexing)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            multi_process=False,
            encode_kwargs={"batch_size": 128},
        )

        # Vector store (child chunks)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        # Parent docstore
        self.docstore = self._load_docstore(self.docstore_pkl)

        # Splitters (clarity / API parity)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size)

        # Base retriever
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

    @staticmethod
    def _load_docstore(pkl_path: str) -> InMemoryStore:
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"Docstore pickle not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            store_data = pickle.load(f)
        ds = InMemoryStore()
        ds.store = store_data  # restore saved dict
        return ds

    @staticmethod
    def _to_json(doc: Document) -> Dict[str, Any]:
        meta = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or ""
        return {
            "case_id": meta.get("case_id"),
            "article_id": meta.get("article_id"),
            "age": meta.get("age"),
            "gender": meta.get("gender"),
            "text": text,
        }

    @staticmethod
    def _retrieve_compat(retriever, query: str, k: int) -> List[Document]:
        # Set top-k when supported
        try:
            retriever.search_kwargs = {"k": max(1, int(k))}
        except Exception:
            pass

        # Runnable retrievers
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)

        # Legacy
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)

        # Private fallback
        if hasattr(retriever, "_get_relevant_documents"):
            return retriever._get_relevant_documents(query)

        raise AttributeError("Retriever lacks invoke/get_relevant_documents/_get_relevant_documents APIs.")

    def _make_ollama_llm(self):
        if _OllamaChat is not None:
            # Chat interface (preferred)
            return _OllamaChat(model=self.rerank_model, temperature=self.rerank_temperature)
        if _OllamaLLM is not None:
            # Legacy LLM interface
            return _OllamaLLM(model=self.rerank_model, temperature=self.rerank_temperature)
        raise RuntimeError(
            "Ollama integration not found. Install either `langchain-ollama` (preferred) or use legacy "
            "`langchain-community` Ollama LLM, and ensure the Ollama daemon is running locally."
        )

    @traceable(name="rerank_cases")
    def rerank_with_llm(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Rerank/Compress retrieved documents using an LLM-driven ContextualCompressionRetriever.

        Notes:
        - This rebuilds a base retriever on the same vectorstore and asks the LLM-based
          compressor to select/condense content most relevant to the query.
        - Returns up to top_n documents (may be fewer if compression drops some).

        Args:
            query: user query text
            documents: list of already-retrieved parent Documents (unused by CCR, but controls k)
            top_n: number of top docs to return after compression
        """
        if not _HAS_COMPRESSION:
            logger.warning("Compression utilities not available; returning original documents.")
            return documents[:top_n]

        try:
            llm = self._make_ollama_llm()
            compressor = LLMChainExtractor.from_llm(llm)

            # Use the count of the already-fetched docs as the CCR search depth
            base_k = max(1, len(documents)) if documents else top_n

            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": base_k})
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )

            # Runnable-style API support
            if hasattr(compression_retriever, "invoke"):
                compressed_docs = compression_retriever.invoke(query)
            elif hasattr(compression_retriever, "get_relevant_documents"):
                compressed_docs = compression_retriever.get_relevant_documents(query)
            else:
                compressed_docs = compression_retriever._get_relevant_documents(query)  # type: ignore

            return (compressed_docs or [])[:top_n]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original documents.")
            return documents[:top_n]

    def retrieve(
        self,
        query: str,
        k: int = 3,
        rerank_top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k parent documents; optionally rerank/compress to top `rerank_top_n`.

        Args:
            query: text query
            k: number of initial results from the ParentDocumentRetriever
            rerank_top_n: if provided, apply LLM-based contextual compression/rerank
                          and return at most this many docs.

        Returns:
            List of JSON dicts with metadata + text.
        """
        initial_docs: List[Document] = self._retrieve_compat(self.retriever, query, k) or []

        if rerank_top_n is not None and rerank_top_n > 0:
            docs = self.rerank_with_llm(query, initial_docs, top_n=rerank_top_n)
        else:
            docs = initial_docs[:k]

        return [self._to_json(doc) for doc in docs]

"""
from medical_multicare_retriever import MedicalMulticareRetriever

retr = MedicalMulticareRetriever(
    persist_dir="./chroma_db_medical_multicare",
    docstore_pkl="./docstore_medical_multicare.pkl",
    embedding_model="pritamdeka/S-PubMedBert-MS-MARCO",
    collection_name="medical_multicare_text",
    rerank_model="llama3.2:3b",
)

# Get 8 candidates, then LLM-compress/rerank to top 3:
results = retr.retrieve("child with cough and fever, likely viral", k=8, rerank_top_n=3)
for i, r in enumerate(results, 1):
    print(f"\n== Result {i} ==")
    for k, v in r.items():
        print(f"{k}: {v}")
"""