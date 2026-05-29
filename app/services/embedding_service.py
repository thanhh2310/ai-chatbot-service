"""
Embedding Service — sử dụng Together AI + intfloat/multilingual-e5-large-instruct.

Model: 1024 chiều, hỗ trợ đa ngôn ngữ (tiếng Việt tốt), giá $0.02/1M tokens.
Dùng chung cho cả RAG search lẫn Auto-Sync vectorization.
"""
import logging
import math
import re
from together import Together
from app.utils.config import Config

logger = logging.getLogger(__name__)

_MODEL = "intfloat/multilingual-e5-large-instruct"
_MAX_EMBED_TOKENS = 250
_CHUNK_OVERLAP_TOKENS = 40
_MAX_SAFE_CHARS = 1500

_client = Together(api_key=Config.TOGETHER_API_KEY)


def _estimate_tokens(text: str) -> int:
    """Ước lượng token đủ bảo thủ cho multilingual text, tránh vượt limit 512."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _split_oversized_segment(segment: str, max_tokens: int) -> list[str]:
    tokens = re.findall(r"\w+|[^\w\s]", segment, flags=re.UNICODE)
    if len(tokens) <= max_tokens:
        return [segment.strip()]

    chunks = []
    step = max(max_tokens - _CHUNK_OVERLAP_TOKENS, 1)
    for start in range(0, len(tokens), step):
        token_chunk = tokens[start:start + max_tokens]
        if not token_chunk:
            continue
        chunks.append(" ".join(token_chunk).strip())
    return chunks


def chunk_text_for_embedding(text: str, max_tokens: int = _MAX_EMBED_TOKENS) -> list[str]:
    """
    Tách text dài thành các chunk dưới giới hạn token của embedding model.
    Ưu tiên cắt theo câu/dấu xuống dòng để giữ ngữ nghĩa, fallback sang token window.
    """
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return []
    if _estimate_tokens(clean) <= max_tokens:
        return [clean]

    segments = re.split(r"(?<=[.!?。！？])\s+|\n+", clean)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        segment_tokens = _estimate_tokens(segment)
        if segment_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            chunks.extend(_split_oversized_segment(segment, max_tokens))
            continue

        if current and current_tokens + segment_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            overlap_text = _build_overlap_text(current)
            current = [overlap_text, segment] if overlap_text else [segment]
            current_tokens = _estimate_tokens(" ".join(current))
        else:
            current.append(segment)
            current_tokens += segment_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _build_overlap_text(parts: list[str]) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", " ".join(parts), flags=re.UNICODE)
    if not tokens:
        return ""
    return " ".join(tokens[-_CHUNK_OVERLAP_TOKENS:]).strip()


def embed_query(text: str) -> list[float]:
    """Embed một chuỗi text, trả về vector 1024 chiều."""
    response = _client.embeddings.create(model=_MODEL, input=text)
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed nhiều chuỗi text cùng lúc (batch), trả về list các vector."""
    response = _client.embeddings.create(model=_MODEL, input=texts)
    # Sắp xếp theo index để đảm bảo thứ tự trả về đúng
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def embed_long_texts(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """
    Embed danh sách text có thể dài quá 512 token.
    Mỗi text được chunk, embed từng chunk, rồi gộp vector trung bình có trọng số token.
    """
    flattened_chunks: list[str] = []
    chunk_meta: list[tuple[int, int]] = []

    for text_index, text in enumerate(texts):
        chunks = chunk_text_for_embedding(text)
        if not chunks:
            chunks = [""]
        for chunk in chunks:
            safe_chunk = chunk[:_MAX_SAFE_CHARS] if len(chunk) > _MAX_SAFE_CHARS else chunk
            flattened_chunks.append(safe_chunk)
            chunk_meta.append((text_index, max(_estimate_tokens(safe_chunk), 1)))

    logger.info(
        "Embedding chunking: %s product texts -> %s chunks; max chunk tokens=%s",
        len(texts),
        len(flattened_chunks),
        max((_estimate_tokens(chunk) for chunk in flattened_chunks), default=0),
    )

    chunk_vectors: list[list[float]] = []
    for start in range(0, len(flattened_chunks), batch_size):
        batch = flattened_chunks[start:start + batch_size]
        chunk_vectors.extend(embed_batch(batch))

    grouped_vectors: list[list[tuple[list[float], int]]] = [[] for _ in texts]
    for vector, (text_index, token_weight) in zip(chunk_vectors, chunk_meta):
        grouped_vectors[text_index].append((vector, token_weight))

    return [_weighted_average_vectors(items) for items in grouped_vectors]


def _weighted_average_vectors(items: list[tuple[list[float], int]]) -> list[float]:
    if not items:
        return []
    dim = len(items[0][0])
    totals = [0.0] * dim
    total_weight = 0

    for vector, weight in items:
        total_weight += weight
        for i, value in enumerate(vector):
            totals[i] += float(value) * weight

    averaged = [value / max(total_weight, 1) for value in totals]
    norm = math.sqrt(sum(value * value for value in averaged))
    if norm == 0:
        return averaged
    return [value / norm for value in averaged]
