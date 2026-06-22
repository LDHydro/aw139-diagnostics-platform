#!/usr/bin/env python3
"""
Build the offline corpus (corpus.json) for the AW139 Diagnostics iPad app.

The server-side embeddings.json was built with OpenAI text-embedding-3-small,
which CANNOT run on-device. For a fully offline iPad app, the document corpus
must be re-embedded with the SAME local model the app uses at query time.

This script reads the existing embeddings.json (we only need its `doc_path`
and `text` fields) and re-embeds every document with a local sentence-transformer
model, producing corpus.json:

    [
      {
        "id": "0",
        "doc_path": "dmc-39-A-24-32-00-00A-320A-A.xml",
        "ata": "24",
        "text": "...",
        "embedding": [0.0123, ...]   # normalized, model-specific dimensionality
      },
      ...
    ]

IMPORTANT: keep the model here in sync with MLXEmbedder.swift (default:
BAAI/bge-small-en-v1.5, 384-dim). If you change one, change the other.

Usage:
    pip install sentence-transformers
    python build_offline_corpus.py \
        --input ../../embeddings.json \
        --output corpus.json \
        --model BAAI/bge-small-en-v1.5

Then transfer corpus.json onto the iPad (Finder → iPad → Files → AW139 Diagnostics).
"""

import argparse
import json
import os
import re
import sys

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DMC_PATTERN = re.compile(r"(\d{2}-[A-Z]-\d{2}-\d{2}-\d{2}-\d{2}[A-Z]-\d{3}[A-Z]?-[A-Z])")


def extract_ata(doc_path: str) -> str:
    """Best-effort ATA chapter from a DMC filename (the 3rd DMC segment)."""
    filename = os.path.basename(doc_path)
    m = DMC_PATTERN.search(filename)
    if m:
        parts = m.group(1).split("-")
        if len(parts) >= 3:
            return parts[2]
    return ""


def load_source(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"Expected a JSON array in {path}")
    docs = []
    for i, d in enumerate(data):
        text = (d.get("text") or "").strip()
        doc_path = d.get("doc_path") or d.get("docPath") or f"doc-{i}"
        if not text:
            continue
        docs.append({"id": str(i), "doc_path": doc_path, "text": text})
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, help="Path to server embeddings.json")
    ap.add_argument("--output", "-o", default="corpus.json", help="Output corpus.json")
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, help="sentence-transformers model id")
    ap.add_argument("--max-chars", type=int, default=3000,
                    help="Truncate document text to this many chars before embedding")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Install dependencies first: pip install sentence-transformers")

    print(f"Loading source documents from {args.input} …")
    docs = load_source(args.input)
    print(f"  {len(docs)} documents with text")

    print(f"Loading embedding model: {args.model} …")
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dimension: {dim}")

    texts = [d["text"][: args.max_chars] for d in docs]
    print("Embedding (this can take a while) …")
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,   # cosine-ready; matches app-side normalization
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    out = []
    for d, vec in zip(docs, vectors):
        out.append({
            "id": d["id"],
            "doc_path": d["doc_path"],
            "ata": extract_ata(d["doc_path"]),
            "text": d["text"][: args.max_chars],
            "embedding": [round(float(x), 6) for x in vec],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {len(out)} documents to {args.output} ({size_mb:.1f} MB, {dim}-dim)")
    print("Transfer this file onto the iPad: Finder → iPad → Files → AW139 Diagnostics.")


if __name__ == "__main__":
    main()
