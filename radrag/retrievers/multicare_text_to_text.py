# medical_multicare_retriever.py
# pip install "langchain>=0.2" langchain-community chromadb langchain-text-splitters huggingface_hub
# Optional (for reranking): pip install langchain-ollama langsmith

from __future__ import annotations
import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
try:
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain_classic.retrievers import ParentDocumentRetriever

# Compression / Reranking (optional)
try:
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    from langchain_classic.retrievers import ContextualCompressionRetriever
    _HAS_COMPRESSION = True
except Exception:
    _HAS_COMPRESSION = False

# Ollama LLM (prefer Chat, fallback to legacy LLM)
_OllamaChat = None
_OllamaLLM = None
try:
    from langchain_ollama import ChatOllama as _OllamaChat
except Exception:
    try:
        from langchain_community.llms import Ollama as _OllamaLLM
    except Exception:
        pass

# LangSmith trace decorator (optional / safe no-op fallback)
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
    Parent-doc retriever over a persisted Chroma store + pickled InMemoryStore.

    Returns a list of dicts:
      {
        "case_id": str | None,
        "article_id": str | None,
        "age": int | str | None,
        "gender": str | None,
        "text": str,            # FULL parent text
        "score": float | None,  # normalized similarity in [0..1], higher is better
        "distance": float | None
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

        # Optional HF login
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
                pass  # not fatal

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            multi_process=False,
            encode_kwargs={"batch_size": 128},
        )

        # Vector store for child chunks
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        # Try to detect metric space from collection metadata (for debugging)
        self._metric = "cosine"
        try:
            coll = getattr(self.vectorstore, "_collection", None)
            if coll and getattr(coll, "metadata", None):
                space = coll.metadata.get("hnsw:space")
                if space in {"cosine", "l2", "ip"}:
                    self._metric = space
        except Exception:
            pass

        logger.info(f"[MedicalMulticareRetriever] Using Chroma metric space: {self._metric}")

        # Parent docstore
        self.docstore = self._load_docstore(self.docstore_pkl)

        # Splitters (to match ParentDocumentRetriever API; not strictly needed for querying)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size)

        # Base parent retriever (compat layer; we’ll also query vectorstore directly for scores)
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

    # ---------------------- low-level helpers ----------------------

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
    def _parent_id_from_meta(meta: Dict[str, Any]) -> Optional[str]:
        # Common keys used by ParentDocumentRetriever / loaders
        for k in ("doc_id", "parent_id", "source", "document_id"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None

    def _fetch_parent_doc(self, parent_id: str) -> Optional[Document]:
        try:
            res = self.docstore.mget([parent_id])
            if res and res[0]:
                return res[0]
        except Exception:
            pass
        return None

    def _to_parent_docs_preserving_order(self, docs: List[Document]) -> List[Document]:
        """Map a list of chunk docs -> unique parent docs in the same order."""
        seen = set()
        ordered_parents: List[Document] = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            pid = self._parent_id_from_meta(meta)
            if not pid or pid in seen:
                continue
            parent = self._fetch_parent_doc(pid)
            if parent:
                seen.add(pid)
                ordered_parents.append(parent)
        return ordered_parents

    @staticmethod
    def _to_json(doc: Document, score: Optional[float] = None, distance: Optional[float] = None) -> Dict[str, Any]:
        meta = getattr(doc, "metadata", {}) or {}
        full_text = getattr(doc, "page_content", "") or ""
        matched_text = meta.get("_best_child_text") or full_text

        return {
            "case_id": meta.get("case_id"),
            "article_id": meta.get("article_id"),
            "age": meta.get("age"),
            "gender": meta.get("gender"),
            "text": full_text,         # full parent
            "matched_text": matched_text,  # child text score is based on
            "score": score,            # float in [0,1]
            "distance": distance,
        }


    def _make_ollama_llm(self):
        if _OllamaChat is not None:
            return _OllamaChat(model=self.rerank_model, temperature=self.rerank_temperature)
        if _OllamaLLM is not None:
            return _OllamaLLM(model=self.rerank_model, temperature=self.rerank_temperature)
        raise RuntimeError(
            "Ollama integration not found. Install `langchain-ollama` or use legacy "
            "`langchain-community` Ollama, and ensure Ollama is running locally."
        )

    # ---------------------- retrieval with scores ----------------------

    def _similarity_search_children_with_scores(
        self, query: str, k: int
    ) -> List[Tuple[Document, Optional[float], Optional[float]]]:
        """
        Use similarity_search_with_score to get raw distances from Chroma,
        then convert distance -> similarity in (0,1].

        Returns list of (child_doc, sim01, distance).
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k) or []
        except Exception as e:
            logger.warning(f"similarity_search_with_score failed: {e}")
            results = []

        out: List[Tuple[Document, Optional[float], Optional[float]]] = []

        for doc, dist in results:
            try:
                dist_f = float(dist)  # handles numpy types too
            except Exception:
                dist_f = None

            if dist_f is None:
                out.append((doc, None, None))
                continue

            # Generic distance -> similarity:
            #   0 distance  -> sim 1.0
            #   1 distance  -> sim 0.5
            #   4 distance  -> sim 0.2
            sim01 = 1.0 / (1.0 + dist_f)
            sim01 = float(sim01)
            sim01 = max(0.0, min(1.0, sim01))  # safety clamp

            logger.debug(
                f"[MedicalMulticareRetriever] metric={self._metric}, "
                f"raw_dist={dist_f:.6f}, sim={sim01:.6f}"
            )
            out.append((doc, sim01, dist_f))

        return out

    def _aggregate_parent_scores(
        self, child_results: List[Tuple[Document, Optional[float], Optional[float]]]
    ) -> List[Tuple[Document, Optional[float], Optional[float]]]:
        parent_best_sim: Dict[str, float] = {}
        parent_best_dist: Dict[str, float] = {}
        parent_first_order: List[str] = []
        parent_doc_cache: Dict[str, Document] = {}

        for child_doc, sim, dist in child_results:
            meta = getattr(child_doc, "metadata", {}) or {}
            pid = self._parent_id_from_meta(meta)
            if not pid:
                continue

            if pid not in parent_first_order:
                parent_first_order.append(pid)

            parent_doc = parent_doc_cache.get(pid)
            if not parent_doc:
                parent = self._fetch_parent_doc(pid)
                if parent:
                    parent_doc_cache[pid] = parent
                    parent_doc = parent

            if sim is not None:
                if (pid not in parent_best_sim) or (sim > parent_best_sim[pid]):
                    parent_best_sim[pid] = sim
                    parent_best_dist[pid] = float(dist) if dist is not None else None

                    if parent_doc is not None:
                        best_child_text = getattr(child_doc, "page_content", "") or ""
                        if not hasattr(parent_doc, "metadata") or parent_doc.metadata is None:
                            parent_doc.metadata = {}
                        parent_doc.metadata["_best_child_text"] = best_child_text

        aggregated: List[Tuple[Document, Optional[float], Optional[float]]] = []
        for pid in parent_first_order:
            parent_doc = parent_doc_cache.get(pid)
            if not parent_doc:
                continue
            aggregated.append(
                (
                    parent_doc,
                    parent_best_sim.get(pid),
                    parent_best_dist.get(pid),
                )
            )
        return aggregated


    # ---------------------- LLM re-rank ----------------------

    @traceable(name="rerank_cases")
    def rerank_with_llm(
        self, query: str, seed_docs: List[Document], top_n: int = 5
    ) -> List[Document]:
        """
        LLM-driven contextual compression / re-ranking, then mapped to full parents.
        Returns full parent Documents (no scores are generated here).
        """
        if not _HAS_COMPRESSION:
            logger.warning("Compression utilities not available; returning original documents.")
            return self._to_parent_docs_preserving_order(seed_docs)[:top_n]

        try:
            llm = self._make_ollama_llm()
            compressor = LLMChainExtractor.from_llm(llm)

            base_k = max(1, len(seed_docs)) if seed_docs else top_n
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": base_k})
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )

            if hasattr(compression_retriever, "invoke"):
                compressed_docs = compression_retriever.invoke(query)
            elif hasattr(compression_retriever, "get_relevant_documents"):
                compressed_docs = compression_retriever.get_relevant_documents(query)
            else:
                compressed_docs = compression_retriever._get_relevant_documents(query)  # type: ignore

            parents = self._to_parent_docs_preserving_order(compressed_docs or [])
            if not parents:
                parents = self._to_parent_docs_preserving_order(seed_docs)[:top_n]
            return parents[:top_n]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original parents.")
            return self._to_parent_docs_preserving_order(seed_docs)[:top_n]

    # ---------------------- public API ----------------------

    def retrieve(
        self,
        query: str,
        k: int = 3,
        rerank_top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k parents with normalized similarity scores; optionally LLM re-rank
        (affects order only) and still return full parent texts.

        Returns: List[dict] with keys:
          - case_id, article_id, age, gender, text, matched_text, score (0..1), distance
        """
        # 1) First pass: get child chunks WITH distances -> normalized similarity in [0,1]
        child_with_scores = self._similarity_search_children_with_scores(
            query, k=max(1, int(k))
        )

        # 2) Aggregate by parent (full parent text + max similarity, paired distance)
        parents_with_scores = self._aggregate_parent_scores(child_with_scores)
        if not parents_with_scores:
            return []

        # Helper: stable key to identify a parent across steps
        def parent_key(doc: Document) -> str:
            meta = getattr(doc, "metadata", {}) or {}
            return str(
                meta.get("case_id")
                or meta.get("article_id")
                or self._parent_id_from_meta(meta)
                or id(doc)  # fallback to in-memory id
            )

        # 3) Optional: LLM-based contextual compression/re-ranking (order only)
        if rerank_top_n is not None and rerank_top_n > 0:
            # map from key -> score/dist
            score_map: Dict[str, Optional[float]] = {}
            dist_map: Dict[str, Optional[float]] = {}

            for p_doc, sim, dist in parents_with_scores:
                key = parent_key(p_doc)
                score_map[key] = sim
                dist_map[key] = dist

            seed_parents = [p for p, _, _ in parents_with_scores]
            reranked_parents = self.rerank_with_llm(query, seed_parents, top_n=rerank_top_n)

            final_pairs: List[Tuple[Document, Optional[float], Optional[float]]] = []
            for p in reranked_parents:
                key = parent_key(p)
                final_pairs.append((p, score_map.get(key), dist_map.get(key)))
        else:
            final_pairs = parents_with_scores[:k]

        # 4) JSON out (full parent text + normalized similarity + distance)
        return [self._to_json(doc, score=score, distance=dist) for doc, score, dist in final_pairs]