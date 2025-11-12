#!/usr/bin/env python3
# pip install chromadb transformers pillow huggingface_hub torch tqdm sentencepiece

import os
import json
import argparse
from pathlib import Path
from shutil import copy2
from typing import List, Dict, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
import chromadb
from huggingface_hub import login
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from transformers import AutoModel as HFModel


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def l2_normalize(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, p=2, dim=-1)


# -------- Models --------

def load_medsiglip(device: torch.device):
    model = AutoModel.from_pretrained("google/medsiglip-448")
    model = model.float().eval().to(device)  # FP32 for MPS/CUDA consistency
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    return model, processor

def load_biobert(device: torch.device, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFModel.from_pretrained(model_name)
    model = model.eval().to(device)
    return tokenizer, model


# -------- Embedding helpers --------

def embed_image(siglip_model, siglip_processor, image_path: str, device: torch.device) -> List[float]:
    img = Image.open(image_path).convert("RGB")
    inputs = siglip_processor(images=[img], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        feats = siglip_model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = l2_normalize(feats)
        vec = feats[0].detach().cpu().float().numpy().tolist()
    return vec

def embed_text_biobert(biobert_tok, biobert_model, text: str, device: torch.device) -> List[float]:
    t = text if text is not None else ""
    tok = biobert_tok(t, return_tensors="pt", truncation=True, max_length=512)
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        out = biobert_model(**tok)
        cls = out.last_hidden_state[:, 0, :]        # [CLS]
        cls = l2_normalize(cls)
        vec = cls[0].detach().cpu().float().numpy().tolist()
    return vec


# -------- ID helpers --------

def paired_ids(result_id: str) -> Tuple[str, str, str]:
    """
    Given a node id like "<base>:img" or "<base>:txt", return:
      base_id, image_id (base:img), text_id (base:txt)
    """
    base = result_id.rsplit(":", 1)[0] if ":" in result_id else result_id
    return base, f"{base}:img", f"{base}:txt"


def resolve_path(meta_image_path: Optional[str], images_root: Optional[str]) -> Optional[str]:
    if not meta_image_path:
        return None
    p = Path(meta_image_path)
    if p.is_absolute():
        return str(p)
    if images_root:
        return str((Path(images_root) / p).resolve())
    return str(p.resolve())


def main(args):

    # HF login (optional)
    if not args.no_login:
        try:
            login(token=args.hf_token, add_to_git_credential=True) if args.hf_token else login(add_to_git_credential=True)
        except Exception as e:
            print(f"[warn] HF login skipped/failed: {e}")

    # Arg validation
    mode = args.mode.lower()
    if mode not in ("image_only", "text_only", "both"):
        raise ValueError("--mode must be one of: image_only, text_only, both")

    if mode == "image_only" and not args.query_image:
        raise ValueError("--query-image is required in image_only mode")
    if mode == "text_only" and not args.query_text:
        raise ValueError("--query-text is required in text_only mode")
    if mode == "both" and not (args.query_image or args.query_text):
        raise ValueError("--mode both requires at least one of --query-image or --query-text")

    ensure_dir(args.out_dir)
    out_jsonl = args.out_jsonl or str(Path(args.out_dir) / "retrieved.jsonl")

    device = choose_device()
    print(f"[info] Using device: {device}")

    # Load models
    print("[info] Loading MedSigLIP (for image queries)…")
    siglip_model, siglip_processor = load_medsiglip(device)
    print(f"[info] Loading BioBERT (for text queries): {args.biobert_model}")
    biobert_tok, biobert_model = load_biobert(device, args.biobert_model)

    # Build query embedding
    qvec = None
    query_from = None
    if args.query_image and mode in ("image_only", "both"):
        print(f"[info] Embedding query image: {args.query_image}")
        qvec = embed_image(siglip_model, siglip_processor, args.query_image, device)
        query_from = "image"
    elif args.query_text:
        snip = (args.query_text[:80] + "...") if len(args.query_text) > 80 else args.query_text
        print(f"[info] Embedding query text with BioBERT: {snip}")
        qvec = embed_text_biobert(biobert_tok, biobert_model, args.query_text, device)
        query_from = "text"
    else:
        raise ValueError("Provide --query-image for image/both modes or --query-text for text/both modes.")

    # Open DB
    print("[info] Opening Chroma collection…")
    client = chromadb.PersistentClient(path=args.chroma_dir)
    col = client.get_or_create_collection(name=args.collection)

    # Build where filter based on mode
    where_filter: Optional[Dict] = None
    if mode == "image_only":
        where_filter = {"modality": "image"}
    elif mode == "text_only":
        where_filter = {"modality": "text"}
    else:
        where_filter = None  # mixed results

    # Query (with graceful fallback if filter yields nothing)
    print(f"[info] Querying '{args.collection}' top-{args.topk} | mode={mode} | query_from={query_from}")
    def do_query(wf: Optional[Dict]):
        return col.query(
            query_embeddings=[qvec],
            n_results=args.topk,
            where=wf,
            include=["metadatas", "documents", "distances"],
        )

    res = do_query(where_filter)
    ids: List[str] = res.get("ids", [[]])[0] if res else []
    if not ids and where_filter is not None:
        print("[warn] No results with filter; retrying without 'where' (metadata may be missing)…")
        res = do_query(None)

    ids = res.get("ids", [[]])[0] if res else []
    metas: List[Dict] = res.get("metadatas", [[]])[0] if res else []
    docs: List[str] = res.get("documents", [[]])[0] if res else []
    dists: List[float] = res.get("distances", [[]])[0] if res else []

    if not ids:
        print("[warn] No results returned.")
        return

    print("[info] Top results:")
    for rank, (rid, dist) in enumerate(zip(ids, dists), 1):
        m = metas[rank-1] if rank-1 < len(metas) else {}
        print(f"  #{rank}: id={rid}  modality={m.get('modality')}  distance={dist:.6f}")

    # Assemble outputs
    print("[info] Collecting outputs (copy images if found, pair text)…")
    ensure_dir(args.out_dir)
    records = []
    for i, rid in enumerate(tqdm(ids, ncols=100)):
        meta = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else None
        modality = meta.get("modality")
        base_id, image_id, text_id = paired_ids(rid)

        # Get/resolve image_path
        image_path_meta = meta.get("image_path")
        image_path = resolve_path(image_path_meta, args.images_root)

        # If current hit is text and image path missing, try paired image node
        if modality == "text" and not (image_path and os.path.isfile(image_path)):
            iget = col.get(ids=[image_id], include=["metadatas"])
            imeta = (iget.get("metadatas") or [None])[0] if iget else None
            if isinstance(imeta, dict):
                image_path = resolve_path(imeta.get("image_path"), args.images_root)

        # Copy image (optional)
        copied_image = None
        if image_path and os.path.isfile(image_path) and not args.no_copy:
            dst = Path(args.out_dir) / Path(image_path).name
            try:
                copy2(image_path, dst)
                copied_image = str(dst)
            except Exception as e:
                print(f"[warn] Failed to copy {image_path} -> {dst}: {e}")

        # Get paired text
        combined_text = None
        if modality == "text" and doc:
            combined_text = doc
        else:
            tget = col.get(ids=[text_id], include=["metadatas", "documents"])
            tmeta = (tget.get("metadatas") or [None])[0] if tget else None
            tdoc = (tget.get("documents") or [None])[0] if tget else None
            combined_text = tdoc or (tmeta.get("combined") if isinstance(tmeta, dict) else None)

        records.append({
            "id": base_id,
            "returned_node_id": rid,
            "returned_modality": modality,
            "distance": dists[i] if i < len(dists) else None,
            "retrieved_image": image_path,
            "copied_image": copied_image,
            "combined": combined_text,
            "image_meta": meta if modality == "image" else None,
        })

    # Write JSONL
    print(f"[info] Writing JSONL -> {out_jsonl}")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[done] Retrieval completed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Unified retrieval from a single Chroma collection. Uses MedSigLIP for image queries and BioBERT for text queries."
    )
    ap.add_argument("--mode", required=True, choices=["image_only", "text_only", "both"], help="Retrieval mode.")
    ap.add_argument("--query-image", dest="query_image", default=None, help="Path to the query image.")
    ap.add_argument("--query-text", dest="query_text", default=None, help="Query text (for text or mixed modes).")
    ap.add_argument("--chroma-dir", required=True, help="ChromaDB directory (single DB).")
    ap.add_argument("--collection", default="rexgrad_unified", help="Collection name.")
    ap.add_argument("--topk", type=int, default=5, help="Number of nearest neighbors to retrieve.")
    ap.add_argument("--out-dir", required=True, help="Folder to copy retrieved images to.")
    ap.add_argument("--out-jsonl", default=None, help="Path to write JSONL (default: <out-dir>/retrieved.jsonl)")
    ap.add_argument("--images-root", default="", help="If image_path in metadata is relative, resolve using this root.")
    ap.add_argument("--no-copy", action="store_true", help="Do not copy retrieved images to out-dir.")

    ap.add_argument("--biobert-model", default="dmis-lab/biobert-v1.1",
                    help="Hugging Face ID for BioBERT used for text embeddings.")

    ap.add_argument("--hf-token", default=None, help="HF token (optional).")
    ap.add_argument("--no-login", action="store_true", help="Skip Hugging Face login.")
    args = ap.parse_args()
    main(args)
