# backend/search_utils.py

import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# Paths
BASE       = Path(__file__).parent
INDEX_PATH = BASE / "data" / "faiss_index.faiss"
META_PATH  = BASE / "data" / "faiss_meta.json"

# Embedder & dimensions
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DIM         = EMBED_MODEL.get_sentence_embedding_dimension()

# Load or init FAISS index
if INDEX_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
else:
    index = faiss.IndexFlatL2(DIM)

# Load or init metadata
if META_PATH.exists():
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
else:
    metadata = []

def save_index():
    """Ghi index và metadata ra đĩa."""
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

def chunk_text(text, chunk_size=1000, overlap=200):
    """Chia text thành các chunk ~chunk_size từ với overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def embed_and_index(chunks, file_name, field):
    """Tính embedding và thêm vào index, cập nhật metadata."""
    embs = EMBED_MODEL.encode(chunks, convert_to_numpy=True)
    start_id = index.ntotal
    index.add(embs)
    for i, chunk in enumerate(chunks):
        metadata.append({
            "id": start_id + i,
            "file": file_name,
            "field": field,
            "chunk": chunk
        })
    save_index()

def search(query, top_k=5):
    """Trả về top_k kết quả semantic search."""
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        item = metadata[idx]
        results.append({
            "file": item["file"],
            "field": item["field"],
            "chunk": item["chunk"],
            "score": float(dist)
        })
    return results
