import re
import unicodedata
from sqlalchemy import text


def filter_products_mentioned_in_reply(reply: str, products: list[dict]) -> list[dict]:
    """
    Keep only products that the final LLM answer actually mentions.

    Matching is intentionally conservative:
    - Product id in markdown/API links or "ID <id>"
    - Slug mention
    - Exact normalized product name mention
    - Strong token overlap for shortened product names
    """
    if not reply or not products:
        return []

    normalized_reply = _normalize(reply)
    mentioned_ids = _mentioned_ids(reply)
    result = []
    seen = set()

    for product in products:
        product_id = int(product["id"])
        if product_id in seen:
            continue
        if product_id in mentioned_ids or _product_name_mentioned(normalized_reply, product):
            result.append(product)
            seen.add(product_id)

    return result


def filter_product_ids_mentioned_in_reply(conn, reply: str, product_ids: list[int]) -> list[int]:
    if not reply or not product_ids:
        return []

    rows = conn.execute(text("""
        SELECT id, name, slug
        FROM products
        WHERE id = ANY(:pids)
    """), {"pids": product_ids}).mappings().all()

    by_id = {row["id"]: dict(row) for row in rows}
    ordered_products = [by_id[pid] for pid in product_ids if pid in by_id]
    filtered = filter_products_mentioned_in_reply(reply, ordered_products)
    return [int(product["id"]) for product in filtered]


def _mentioned_ids(reply: str) -> set[int]:
    patterns = [
        r"/api/products/(\d+)",
        r"/products/(\d+)",
        r"\bID\s*[:#-]?\s*(\d+)\b",
        r"\bsản phẩm\s*[:#-]?\s*(\d+)\b",
    ]
    ids: set[int] = set()
    for pattern in patterns:
        for match in re.findall(pattern, reply, flags=re.IGNORECASE | re.UNICODE):
            try:
                ids.add(int(match))
            except ValueError:
                pass
    return ids


def _product_name_mentioned(normalized_reply: str, product: dict) -> bool:
    name = _normalize(str(product.get("name") or ""))
    slug = _normalize(str(product.get("slug") or ""))
    if name and name in normalized_reply:
        return True
    if slug and slug in normalized_reply:
        return True

    name_tokens = _meaningful_tokens(name)
    if len(name_tokens) < 3:
        return False

    matched = sum(1 for token in name_tokens if _token_in_text(token, normalized_reply))
    overlap = matched / max(len(name_tokens), 1)
    return matched >= 3 and overlap >= 0.65


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value or "")
    without_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    without_accents = without_accents.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_accents.lower()).strip()


def _meaningful_tokens(value: str) -> list[str]:
    stopwords = {
        "ao", "quan", "nam", "nu", "the", "thao", "san", "pham", "coolmate",
        "size", "mau", "va", "cho", "cua", "voi", "hang",
    }
    tokens = re.findall(r"[a-z0-9]+", value)
    result = []
    for token in tokens:
        if len(token) < 2 or token in stopwords:
            continue
        if token not in result:
            result.append(token)
    return result


def _token_in_text(token: str, text_value: str) -> bool:
    return re.search(rf"(?:^|\W){re.escape(token)}(?:$|\W)", text_value, flags=re.UNICODE) is not None
