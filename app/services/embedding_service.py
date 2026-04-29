"""
Embedding Service — sử dụng Together AI + intfloat/multilingual-e5-large-instruct.

Model: 1024 chiều, hỗ trợ đa ngôn ngữ (tiếng Việt tốt), giá $0.02/1M tokens.
Dùng chung cho cả RAG search lẫn Auto-Sync vectorization.
"""
import logging
from together import Together
from app.utils.config import Config

logger = logging.getLogger(__name__)

_MODEL = "intfloat/multilingual-e5-large-instruct"

_client = Together(api_key=Config.TOGETHER_API_KEY)


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
