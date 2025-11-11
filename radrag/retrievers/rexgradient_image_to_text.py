# rexgradient_retriever.py
# pip install chromadb transformers pillow torch sentencepiece huggingface_hub

from __future__ import annotations
from typing import Optional, Dict, Any, List

import os
import torch
from PIL import Image
import chromadb
from transformers import AutoProcessor, AutoModel

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


def _maybe_hf_login(
    hf_cache_dir: Optional[str],
    hf_token: Optional[str] = None,
    quiet: bool = True,
) -> None:
    """Best-effort Hugging Face login & optional cache setup."""
    # Respect user-provided cache dir
    if hf_cache_dir:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

    # Token precedence: explicit arg → HUGGINGFACE_HUB_TOKEN → HF_TOKEN
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
    • Returns only the combined text (string) or None.
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

    def retrieve(self, query_image_full_path: str) -> Optional[str]:
        if not os.path.isfile(query_image_full_path):
            raise FileNotFoundError(f"Query image not found: {query_image_full_path}")

        qvec = _embed_image(self.model, self.processor, query_image_full_path, self.device)

        def _do_query(where_filter: Optional[Dict[str, Any]]):
            # FIX: do not include "ids" (it's always returned)
            return self.col.query(
                query_embeddings=[qvec],
                n_results=1,
                where=where_filter,
                include=["metadatas", "documents", "distances"],
            )

        # Prefer image nodes; if none, allow any
        res = _do_query({"modality": "image"})
        ids = res.get("ids", [[]])[0] if res else []
        if not ids:
            res = _do_query(None)
            ids = res.get("ids", [[]])[0] if res else []
            if not ids:
                return None

        rid = ids[0]
        metas = res.get("metadatas", [[]])[0] if res else []
        docs = res.get("documents", [[]])[0] if res else []
        meta = metas[0] if metas else {}
        doc = docs[0] if docs else None
        modality = meta.get("modality")

        # If top hit is already a text node with a document, return it
        if modality == "text" and isinstance(doc, str) and doc.strip():
            return doc

        # Otherwise, fetch paired text node
        _, _, text_id = _paired_ids(rid)
        try:
            tget = self.col.get(ids=[text_id], include=["metadatas", "documents"])
            tmeta = (tget.get("metadatas") or [None])[0] if tget else None
            tdoc = (tget.get("documents") or [None])[0] if tget else None
            if isinstance(tdoc, str) and tdoc.strip():
                return tdoc
            if isinstance(tmeta, dict):
                fallback = tmeta.get("combined") or tmeta.get("combined_text")
                if isinstance(fallback, str) and fallback.strip():
                    return fallback
        except Exception:
            pass

        # Final fallback: some image nodes may carry combined text in metadata
        fallback_img = meta.get("combined") or meta.get("combined_text")
        if isinstance(fallback_img, str) and fallback_img.strip():
            return fallback_img

        return None