#!/usr/bin/env python3

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from shutil import copy2

import torch
from PIL import Image
from huggingface_hub import login
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModel as HFModel

import chromadb


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size_bytes(path: str) -> int:
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    for fp in p.rglob("*"):
        if fp.is_file():
            total += fp.stat().st_size
    return total


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def main(args):

    # HF login (optional)
    if not args.no_login:
        print("[info] Logging in to Hugging Face Hub…")
        try:
            login(token=args.hf_token, add_to_git_credential=True) if args.hf_token else login(add_to_git_credential=True)
        except Exception as e:
            print(f"[warn] HF login skipped or failed: {e}")

    # Device
    device = choose_device()
    print(f"[info] Using device: {device}")

    # Load models
    print("[info] Loading MedSigLIP…")
    img_model = AutoModel.from_pretrained("google/medsiglip-448")
    img_processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    img_model.eval().to(device)

    print("[info] Loading BioBERT…")
    txt_model_name = "dmis-lab/biobert-v1.1"
    txt_tokenizer = AutoTokenizer.from_pretrained(txt_model_name)
    txt_model: HFModel = HFModel.from_pretrained(txt_model_name)
    txt_model.eval().to(device)

    # Prepare TWO Chroma clients (two DBs)
    ensure_dir(args.images_chroma_dir)
    ensure_dir(args.text_chroma_dir)
    print("[info] Initializing ChromaDB (images)…")
    img_client = chromadb.PersistentClient(path=args.images_chroma_dir)
    print("[info] Initializing ChromaDB (text)…")
    txt_client = chromadb.PersistentClient(path=args.text_chroma_dir)

    img_collection = img_client.get_or_create_collection(name=args.images_collection)
    txt_collection = txt_client.get_or_create_collection(name=args.text_collection)

    # Load & shuffle data
    print("[info] Loading JSONL…")
    rows = load_jsonl(args.jsonl)
    if not rows:
        raise RuntimeError("No valid rows found in JSONL.")
    random.seed(args.seed)
    random.shuffle(rows)

    # Build the valid list
    valid = []
    img_root = Path(args.images_root)
    for r in rows:
        rel = (r.get("image_path") or "").strip()
        if not rel:
            continue
        src_img = img_root / Path(rel).name  # use basename only
        if src_img.is_file():
            valid.append({
                "id": os.path.splitext(Path(rel).name)[0],  # shared ID (without extension)
                "src_image": str(src_img),
                "combined": r.get("combined", "") or "",
            })
        # Respect --limit only when not --full
        if not args.full and len(valid) >= args.limit:
            break

    if not valid:
        raise RuntimeError("No valid image files found matching basenames in --images-root.")

    if not args.full and len(valid) < args.limit:
        print(f"[warn] Only {len(valid)} valid samples found (requested {args.limit}). Proceeding.")

    print(f"[info] Will index {len(valid)} samples with batch-size={args.batch_size}.")

    # Baseline sizes for both DBs
    img_size_before = dir_size_bytes(args.images_chroma_dir)
    txt_size_before = dir_size_bytes(args.text_chroma_dir)
    print(f"[info] Images DB initial size: {human_bytes(img_size_before)}")
    print(f"[info] Text   DB initial size: {human_bytes(txt_size_before)}")

    # Batch insert (measuring per-batch growth)
    print("[info] Encoding & inserting in batches…")
    per_batch_img_growth = []
    per_batch_txt_growth = []
    curr_img_size = img_size_before
    curr_txt_size = txt_size_before

    subset_records = []
    total = len(valid)
    bs = max(1, int(args.batch_size))
    num_batches = (total + bs - 1) // bs

    pbar = tqdm(range(num_batches), ncols=120)
    for bi in pbar:
        start = bi * bs
        end = min(start + bs, total)
        batch = valid[start:end]
        pbar.set_description_str(f"[batch {bi+1}/{num_batches} | {end-start} items]")

        # --- Prepare images & texts ---
        pil_images = []
        texts = []
        ids = []
        metas = []
        docs = []
        # For JSONL/copy
        dst_paths_for_jsonl = []

        for r in batch:
            ids.append(r["id"])
            texts.append(r["combined"] or "")
            metas.append({
                "image_path": r["src_image"],
                "combined": r["combined"] or "",
            })
            docs.append(r["combined"] or "")
            # default path for JSONL (may be replaced if copying)
            dst_paths_for_jsonl.append(r["src_image"])
            try:
                pil_images.append(Image.open(r["src_image"]).convert("RGB"))
            except Exception as e:
                print(f"[warn] Skipping {r['id']} (image open failed): {e}")
                # keep alignment by adding a dummy small black image to preserve batching
                pil_images.append(Image.new("RGB", (8, 8)))

        # --- IMAGE EMBEDDINGS (batched) ---
        img_inputs = img_processor(
            text=["a photo of a chest x-ray"] * len(pil_images),
            images=pil_images,
            padding="max_length",
            return_tensors="pt"
        )
        img_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in img_inputs.items()}
        with torch.no_grad():
            img_out = img_model(**img_inputs)
            # (B, D)
            img_vecs = img_out.image_embeds.detach().cpu().float().numpy().tolist()

        # --- TEXT EMBEDDINGS (batched) ---
        tok = txt_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            tout = txt_model(**tok)
            cls = tout.last_hidden_state[:, 0, :]                    # (B, 768)
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)    # L2 normalize
            txt_vecs = cls.detach().cpu().float().numpy().tolist()

        # --- UPSERT (batched) ---
        img_collection.upsert(ids=ids, embeddings=img_vecs, metadatas=metas, documents=docs)
        txt_collection.upsert(ids=ids, embeddings=txt_vecs, metadatas=metas, documents=docs)

        # --- Measure growth (per-batch) ---
        new_img = dir_size_bytes(args.images_chroma_dir)
        new_txt = dir_size_bytes(args.text_chroma_dir)
        inc_img = new_img - curr_img_size
        inc_txt = new_txt - curr_txt_size
        per_batch_img_growth.append(inc_img)
        per_batch_txt_growth.append(inc_txt)
        curr_img_size = new_img
        curr_txt_size = new_txt
        pbar.set_postfix_str(f"+img {human_bytes(inc_img)} | +txt {human_bytes(inc_txt)}")

        # --- Copy images (optional) & collect subset records ---
        if args.copy_images_to:
            ensure_dir(args.copy_images_to)
        for i, r in enumerate(batch):
            dst_path = r["src_image"]
            if args.copy_images_to:
                dst_img_path = Path(args.copy_images_to) / Path(r["src_image"]).name
                try:
                    if args.overwrite or not dst_img_path.exists():
                        copy2(r["src_image"], dst_img_path)
                    dst_path = str(dst_img_path)
                except Exception as e:
                    print(f"[warn] Failed to copy {r['src_image']} -> {dst_img_path}: {e}")
            subset_records.append({
                "id": r["id"],
                "image_path": dst_path,
                "combined": r["combined"] or ""
            })

        # free per-batch PILs ASAP
        del pil_images
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Summary for DBs
    final_img_size = dir_size_bytes(args.images_chroma_dir)
    final_txt_size = dir_size_bytes(args.text_chroma_dir)

    print("\n=== Per-batch growth ===")
    for i, (gi, gt) in enumerate(zip(per_batch_img_growth, per_batch_txt_growth), 1):
        print(f"batch {i:03d}: +img {human_bytes(gi)} | +txt {human_bytes(gt)}")

    print("\n=== Counts ===")
    print(f"Images collection count: {img_collection.count()}")
    print(f"Text   collection count: {txt_collection.count()}")

    print("\n=== Size summary ===")
    print(f"Images DB initial : {human_bytes(img_size_before)}")
    print(f"Images DB final   : {human_bytes(final_img_size)}")
    print(f"Images DB growth  : {human_bytes(final_img_size - img_size_before)}")
    print(f"Text   DB initial : {human_bytes(txt_size_before)}")
    print(f"Text   DB final   : {human_bytes(final_txt_size)}")
    print(f"Text   DB growth  : {human_bytes(final_txt_size - txt_size_before)}")
    print(f"TOTAL growth      : {human_bytes((final_img_size - img_size_before) + (final_txt_size - txt_size_before))}")

    # Write subset JSONL if requested
    if args.out_jsonl:
        out_jsonl_path = Path(args.out_jsonl)
        if out_jsonl_path.exists() and not args.overwrite:
            raise FileExistsError(f"{args.out_jsonl} exists. Pass --overwrite to replace it.")
        ensure_dir(out_jsonl_path.parent)
        print(f"\n[info] Writing subset JSONL -> {args.out_jsonl}")
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for rec in subset_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        copied_note = f" and copied images to {args.copy_images_to}" if args.copy_images_to else ""
        print(f"[done] Wrote JSONL with {len(subset_records)} records{copied_note}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TWO ChromaDBs (images/text) in batches + optional copy subset images + JSONL."
    )
    parser.add_argument("--jsonl", required=True, help="Path to input JSONL file.")
    parser.add_argument("--images-root", required=True, help="Root dir of images; we resolve basename(image_path) here.")

    # Two separate DB dirs
    parser.add_argument("--images-chroma-dir", required=True, help="ChromaDB directory for IMAGE embeddings.")
    parser.add_argument("--text-chroma-dir", required=True, help="ChromaDB directory for TEXT embeddings.")
    parser.add_argument("--images-collection", default="images", help="Collection name for image vectors.")
    parser.add_argument("--text-collection", default="text", help="Collection name for text vectors.")

    # Copy target and JSONL output
    parser.add_argument("--copy-images-to", default="", help="Folder to copy the selected images.")
    parser.add_argument("--out-jsonl", default="", help="Path to write JSONL with ONLY the used (indexed) samples.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing out JSONL and copied files.")

    parser.add_argument("--limit", type=int, default=200, help="Number of samples to index (post-shuffle & validity).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (if models are gated).")
    parser.add_argument("--no-login", action="store_true", help="Skip Hugging Face login.")
    parser.add_argument("--full", action="store_true", help="Index ALL valid rows in the JSONL (ignore --limit).")

    # NEW: batching
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding + upsert.")

    args = parser.parse_args()
    main(args)
