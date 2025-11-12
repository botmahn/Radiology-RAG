# rexgradient_retriever.py
# pip install chromadb transformers pillow torch sentencepiece huggingface_hub

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple

import os
import torch
from PIL import Image
import chromadb
from transformers import AutoProcessor, AutoModel
import numpy as np
# HF login (optional)
try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, p=2, dim=-1)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity for 1D tensors."""
    a = _l2_normalize(a.view(-1))
    b = _l2_normalize(b.view(-1))
    return float(torch.dot(a, b).item())


def _maybe_hf_login(
    hf_cache_dir: Optional[str],
    hf_token: Optional[str] = None,
    quiet: bool = True,
) -> None:
    """Best-effort Hugging Face login & optional cache setup."""
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token or hf_login is None:
        return
    try:
        hf_login(token=token, add_to_git_credential=False)
        if not quiet:
            print("[hf] Login successful (or token already cached).")
    except Exception as e:
        if not quiet:
            print(f"[hf] Login skipped/failed: {e}")


def _load_medsiglip(device: torch.device):
    model = AutoModel.from_pretrained("google/medsiglip-448")
    model = model.float().eval().to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    return model, processor


def _embed_image(model, processor, image_path: str, device: torch.device) -> List[float]:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = _l2_normalize(feats)
        vec = feats[0].detach().cpu().float().numpy().tolist()
    return vec


def _paired_ids(result_id: str):
    """For an id like '<base>:img' or '<base>:txt', return (base, base:img, base:txt)."""
    base = result_id.rsplit(":", 1)[0] if ":" in result_id else result_id
    return base, f"{base}:img", f"{base}:txt"


class RexGradientRetriever:
    """
    Image -> Text retriever for a Chroma collection with paired image/text nodes.

    • Query with a single full-path image.
    • Fixed top-k = 1.
    • Returns combined text (string) or (text, similarity) if return_score=True.

    When return_score=True, similarity is cosine similarity between the query
    image embedding and the top hit's stored embedding (computed locally from
    the returned vector), independent of the index's distance metric.
    """

    def __init__(
        self,
        chroma_dir: str,
        collection: str = "rexgrad_unified",
        device: Optional[torch.device] = None,
        # HF login controls
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        login_on_init: bool = True,
        verbose_login: bool = False,
    ) -> None:
        if login_on_init:
            _maybe_hf_login(hf_cache_dir=hf_cache_dir, hf_token=hf_token, quiet=not verbose_login)

        self.device = device or _choose_device()
        self.model, self.processor = _load_medsiglip(self.device)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.col = self.client.get_or_create_collection(name=collection)

    def retrieve(self, query_image_full_path: str, return_score: bool = False):

        def _safe_len(x) -> int:
            try:
                return len(x)
            except Exception:
                return 0

        def _safe_first(x):
            # returns first element or None, never boolean-tests arrays
            if x is None:
                return None
            if isinstance(x, (list, tuple)) and _safe_len(x) > 0:
                return x[0]
            return None

        def _to_tensor(vec):
            # vec may be list / list[list] / np.ndarray / nested
            if vec is None:
                return None
            if isinstance(vec, np.ndarray):
                return torch.tensor(vec, dtype=torch.float32)
            if isinstance(vec, (list, tuple)):
                # peel nesting like [[...]] or [[[...]]]
                v = vec
                while isinstance(v, (list, tuple)) and _safe_len(v) > 0 and isinstance(v[0], (list, tuple, np.ndarray)):
                    v = v[0]
                return torch.tensor(v, dtype=torch.float32)
            return None

        if not os.path.isfile(query_image_full_path):
            raise FileNotFoundError(f"Query image not found: {query_image_full_path}")

        # embed query
        qvec = _embed_image(self.model, self.processor, query_image_full_path, self.device)
        qvec_t = _to_tensor(qvec)
        if qvec_t is None:
            raise RuntimeError("Failed to build query embedding tensor.")

        def _do_query(where_filter: Optional[Dict[str, Any]]):
            return self.col.query(
                query_embeddings=[qvec],
                n_results=1,
                where=where_filter,
                include=["metadatas", "documents", "embeddings", "distances"],
            )

        # first try only image nodes
        res = _do_query({"modality": "image"})
        ids = _safe_first(res.get("ids")) if isinstance(res, dict) else None
        if not ids or _safe_len(ids) == 0:
            # fallback: any node
            res = _do_query(None)
            ids = _safe_first(res.get("ids")) if isinstance(res, dict) else None
            if not ids or _safe_len(ids) == 0:
                return (None, None) if return_score else None

        rid = ids[0]

        metas_list = _safe_first(res.get("metadatas"))
        docs_list = _safe_first(res.get("documents"))
        embs_list = _safe_first(res.get("embeddings"))
        dists_list = _safe_first(res.get("distances"))

        meta = metas_list[0] if isinstance(metas_list, (list, tuple)) and _safe_len(metas_list) > 0 else {}
        doc  = docs_list[0] if isinstance(docs_list, (list, tuple)) and _safe_len(docs_list) > 0 else None
        remb = embs_list[0] if isinstance(embs_list, (list, tuple, np.ndarray)) and _safe_len(embs_list) > 0 else None
        dist = dists_list[0] if isinstance(dists_list, (list, tuple, np.ndarray)) and _safe_len(dists_list) > 0 else None

        modality = meta.get("modality")

        # similarity (cosine) if we got an embedding back
        sim = None
        rvec_t = _to_tensor(remb)
        if isinstance(rvec_t, torch.Tensor) and rvec_t.numel() > 0:
            def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
                a = torch.nn.functional.normalize(a.view(1, -1), p=2, dim=-1)
                b = torch.nn.functional.normalize(b.view(1, -1), p=2, dim=-1)
                return float((a @ b.T).item())
            try:
                sim = _cosine_sim(qvec_t, rvec_t)
            except Exception as e:
                print(f"[warn] cosine sim failed: {e}")
                sim = None

        # If top hit is already a text node with a document, return it
        if modality == "text" and isinstance(doc, str) and doc.strip():
            return (doc, sim) if return_score else doc

        # Otherwise, fetch paired text node
        base = rid.rsplit(":", 1)[0] if isinstance(rid, str) and ":" in rid else rid
        text_id = f"{base}:txt" if isinstance(base, str) else None
        if text_id:
            try:
                tget = self.col.get(ids=[text_id], include=["metadatas", "documents"])
                tmeta = _safe_first(tget.get("metadatas")) if isinstance(tget, dict) else None
                tdoc  = _safe_first(tget.get("documents")) if isinstance(tget, dict) else None
                tmeta = tmeta if isinstance(tmeta, dict) else {}
                tdoc  = tdoc if isinstance(tdoc, str) else None
                if tdoc and tdoc.strip():
                    return (tdoc, sim) if return_score else tdoc
                fallback = tmeta.get("combined") or tmeta.get("combined_text")
                if isinstance(fallback, str) and fallback.strip():
                    return (fallback, sim) if return_score else fallback
            except Exception as e:
                print(f"[warn] get(text_id) failed: {e}")

        # Final fallback: sometimes the image node itself carries combined text
        fallback_img = meta.get("combined") or meta.get("combined_text")
        if isinstance(fallback_img, str) and fallback_img.strip():
            return (fallback_img, sim) if return_score else fallback_img

        return (None, sim) if return_score else None