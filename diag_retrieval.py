import argparse
import asyncio
import os
import re
import time
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np

# Use httpx & PyPDF2 only if needed for local file; prefer using existing processor for URL
try:
    import httpx  # type: ignore
except Exception:
    httpx = None

# Reuse robust PDF/text pipeline and chunker from the app
from app_groq_ultimate import GroqDocumentProcessor

STOPWORDS = set(
    """
    a an the and or but if when while of to for from in on at by with without into over under as is are was were be been being this that these those then than so such not no nor can could shall should will would may might must do does did done having have has he she it they them we you your our their its i me my mine ours yours theirs his her hers itself themselves ourselves yourself yourselves himself herself
    """.split()
)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%₹$\s]", " ", text)
    toks = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]
    return toks


def build_tfidf(chunks: List[str]) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    """
    Build a TF-IDF matrix for chunks.
    Returns: (vocab, tfidf_matrix[ndocs x nvocab], idf[nvocab])
    """
    tokenized = [tokenize(ch) for ch in chunks]
    vocab: Dict[str, int] = {}
    for toks in tokenized:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    nv = len(vocab)
    nd = len(chunks)
    if nv == 0 or nd == 0:
        return vocab, np.zeros((nd, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Document frequency
    df = np.zeros((nv,), dtype=np.float32)
    for toks in tokenized:
        seen = set()
        for t in toks:
            idx = vocab[t]
            if idx not in seen:
                df[idx] += 1.0
                seen.add(idx)
    # idf with smoothing
    idf = np.log((nd + 1) / (df + 1)) + 1.0

    # TF-IDF matrix
    X = np.zeros((nd, nv), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        tf: Dict[int, float] = {}
        for t in toks:
            idx = vocab[t]
            tf[idx] = tf.get(idx, 0.0) + 1.0
        # log-scaled tf
        for idx, c in tf.items():
            X[i, idx] = (1.0 + np.log(c)) * idf[idx]

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X /= norms
    return vocab, X, idf


def encode_query(question: str, vocab: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    toks = tokenize(question)
    v = np.zeros((len(vocab),), dtype=np.float32)
    tf: Dict[int, float] = {}
    for t in toks:
        if t in vocab:
            idx = vocab[t]
            tf[idx] = tf.get(idx, 0.0) + 1.0
    for idx, c in tf.items():
        v[idx] = (1.0 + np.log(c)) * idf[idx]
    n = np.linalg.norm(v) + 1e-8
    return v / n


def rank_chunks(question: str, chunks: List[str]) -> List[Tuple[int, float]]:
    vocab, X, idf = build_tfidf(chunks)
    if X.shape[1] == 0:
        return []
    qv = encode_query(question, vocab, idf)
    sims = X @ qv  # cosine similarity because rows are normalized
    order = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in order]


async def your_retrieval_function(document_url: str, question: str, top_k: int = 5) -> Tuple[List[str], List[Tuple[int, float]]]:
    """
    Load document via existing processor, chunk it, build a TF-IDF index, and return top_k chunks + scores.
    """
    processor = GroqDocumentProcessor()
    content = await processor._get_clean_document_content(document_url)
    if not content or len(content.strip()) < 50:
        raise RuntimeError("Document content is empty or too short. Check the URL or PDF extraction.")
    chunks = processor._get_or_build_chunks(content)
    if not chunks:
        raise RuntimeError("No chunks built from document content.")
    ranked = rank_chunks(question, chunks)[: top_k]
    top_chunks = [chunks[i] for i, _ in ranked]
    return top_chunks, ranked


async def main():
    parser = argparse.ArgumentParser(description="Retrieval Diagnostic: TF-IDF chunk search for a PDF URL")
    parser.add_argument("--url", required=True, help="PDF URL (e.g., https://.../HDFHLIP23024V072223.pdf)")
    parser.add_argument("--question", required=True, help="Question to retrieve context for")
    parser.add_argument("--top_k", type=int, default=5, help="Top K chunks to log")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"retrieval_diag_{ts}.txt")

    print(f"[DIAG] Starting retrieval diagnostic for: {args.url}")
    print(f"[DIAG] Question: {args.question}")
    t0 = time.time()
    try:
        chunks, scores = await your_retrieval_function(args.url, args.question, args.top_k)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        raise
    dt = (time.time() - t0) * 1000.0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"URL: {args.url}\n")
        f.write(f"Question: {args.question}\n")
        f.write(f"Elapsed: {dt:.1f} ms\n")
        f.write("Top Chunks:\n")
        for idx, (i, s) in enumerate(scores[: args.top_k], 1):
            f.write(f"\n[{idx}] score={s:.4f}\n")
            snippet = chunks[idx - 1].strip()
            f.write(snippet + "\n")

    print(f"[DIAG] Retrieved top {len(chunks)} chunks in {dt:.1f} ms")
    print(f"[DIAG] Log written to: {log_path}")
    # Also echo to console
    for idx, (i, s) in enumerate(scores[: args.top_k], 1):
        print(f"\n[{idx}] score={s:.4f}")
        print(chunks[idx - 1][:400] + ("..." if len(chunks[idx - 1]) > 400 else ""))


if __name__ == "__main__":
    asyncio.run(main())
