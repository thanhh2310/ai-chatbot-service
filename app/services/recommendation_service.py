import logging
from sqlalchemy import text
from app.services.db_service import db
import numpy as np

engine = db.get_engine()
logger = logging.getLogger(__name__)

_SEEDS_PER_SOURCE = 3


def get_recommendations_for_user(user_id: int, limit: int = 6) -> list[int]:
    """
    Gợi ý sản phẩm cá nhân hóa dựa trên lịch sử mua, wishlist, và user_interactions.

    Chiến lược:
      - Seed từ 4 nguồn: orders, wishlists, user_interactions (view/search/cart)
      - Tính vector trọng tâm (centroid) từ embeddings của seed products
      - Tìm sản phẩm tương tự bằng vector similarity
      - Popularity fusion: ưu tiên sản phẩm xuất hiện nhiều lần trong lịch sử
      - Fallback: sản phẩm phổ biến nhất nếu user chưa có dữ liệu
    """
    try:
        with engine.connect() as conn:
            # ── 1. Sản phẩm loại trừ (đã mua, đã wishlist) ───────────────────
            exclude_ids = _get_excluded_ids(conn, user_id)

            # ── 2. Seed products từ 4 nguồn ────────────────────────────────────
            seed_ids = _get_seeds(conn, user_id)

            # ── 3. Fallback: user mới → sản phẩm phổ biến ──────────────────────
            if not seed_ids:
                return _get_popular_products(conn, limit, exclude_ids)

            # ── 4. Embeddings của seed products ────────────────────────────────
            seed_vectors = _get_embeddings(conn, seed_ids)
            if not seed_vectors:
                return _get_popular_products(conn, limit, exclude_ids)

            # ── 5. Tính centroid (vector trung bình có trọng số) ─────────────────
            centroid = _compute_weighted_centroid(conn, user_id, seed_ids, seed_vectors)

            # ── 6. Tìm sản phẩm tương tự + popularity fusion ─────────────────
            candidates = _find_similar_with_popularity(conn, centroid, exclude_ids, limit)

            logger.info(f"✅ Recommendations user={user_id}: {[pid for pid, _ in candidates]}")
            return [pid for pid, _ in candidates]

    except Exception as e:
        logger.error(f"❌ Recommendation lỗi user={user_id}: {e}", exc_info=True)
        return []


# ── PRIVATE HELPERS ──────────────────────────────────────────────────────────

def _get_excluded_ids(conn, user_id: int) -> set[int]:
    """Lấy tất cả SP đã mua + đã wishlist để loại trừ khỏi gợi ý."""
    rows = conn.execute(text("""
        SELECT DISTINCT product_id FROM (
            SELECT ps.product_id
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN product_skus ps ON oi.product_sku_id = ps.id
            WHERE o.user_id = :uid

            UNION

            SELECT product_id FROM wishlists WHERE user_id = :uid
        ) AS excluded
    """), {"uid": user_id}).scalars().all()
    return set(rows)


def _get_seeds(conn, user_id: int) -> list[int]:
    """
    Lấy seed products từ 4 nguồn với trọng số:
      - PURCHASE: weight 5.0
      - ADD_TO_CART: weight 3.0
      - VIEW: weight 1.0
      - SEARCH: weight 0.5
    """
    rows = conn.execute(text("""
        WITH seeds AS (
            -- 1. Từ orders (mua hàng - trọng số cao nhất)
            SELECT DISTINCT ps.product_id, 5.0 AS weight
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN product_skus ps ON oi.product_sku_id = ps.id
            WHERE o.user_id = :uid AND o.order_status != 'CANCELLED'

            UNION ALL

            -- 2. Từ wishlists
            SELECT product_id, 4.0 AS weight
            FROM wishlists
            WHERE user_id = :uid

            UNION ALL

            -- 3. Từ user_interactions (cart)
            SELECT ui.product_id, 3.0 AS weight
            FROM user_interactions ui
            WHERE ui.user_id = :uid AND ui.interaction_type = 'ADD_TO_CART'

            UNION ALL

            -- 4. Từ user_interactions (view)
            SELECT ui.product_id, 1.0 AS weight
            FROM user_interactions ui
            WHERE ui.user_id = :uid AND ui.interaction_type = 'VIEW'

            UNION ALL

            -- 5. Từ user_interactions (search - trọng số thấp nhất)
            SELECT ui.product_id, 0.5 AS weight
            FROM user_interactions ui
            WHERE ui.user_id = :uid AND ui.interaction_type = 'SEARCH'
        )
        SELECT product_id, SUM(weight) AS total_weight
        FROM seeds
        GROUP BY product_id
        ORDER BY total_weight DESC
        LIMIT :limit
    """), {"uid": user_id, "limit": _SEEDS_PER_SOURCE * 4}).mappings().all()
    return [r["product_id"] for r in rows]


def _compute_weighted_centroid(conn, user_id: int, seed_ids: list[int], seed_vectors: dict) -> str:
    """
    Tính centroid có trọng số dựa trên interaction_weight và tần suất xuất hiện.
    """
    # Lấy trọng số của từng seed product
    weights = {}
    rows = conn.execute(text("""
        WITH seeds AS (
            SELECT DISTINCT ps.product_id, 5.0 AS weight
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN product_skus ps ON oi.product_sku_id = ps.id
            WHERE o.user_id = :uid AND o.order_status != 'CANCELLED'

            UNION ALL
            SELECT product_id, 4.0 FROM wishlists WHERE user_id = :uid
            UNION ALL
            SELECT product_id, ui.interaction_weight FROM user_interactions ui WHERE ui.user_id = :uid
        )
        SELECT product_id, SUM(weight) AS total_weight
        FROM seeds
        WHERE product_id = ANY(:pids)
        GROUP BY product_id
    """), {"uid": user_id, "pids": seed_ids}).mappings().all()

    for r in rows:
        weights[r["product_id"]] = float(r["total_weight"])

    # Normalize weights
    total_weight = sum(weights.values()) if weights else 1.0

    vector_arrays = []
    for pid in seed_ids:
        if pid not in seed_vectors:
            continue
        w = weights.get(pid, 1.0) / total_weight
        clean_str = seed_vectors[pid].strip("[]")
        vec_float = [float(x) * w for x in clean_str.split(",")]
        vector_arrays.append(vec_float)

    if not vector_arrays:
        # Fallback: simple average
        for pid in seed_ids:
            if pid in seed_vectors:
                clean_str = seed_vectors[pid].strip("[]")
                vec_float = [float(x) for x in clean_str.split(",")]
                vector_arrays.append(vec_float)
                break

    if not vector_arrays:
        return ""

    centroid = np.mean(vector_arrays, axis=0)
    return f"[{','.join(map(str, centroid))}]"


def _get_embeddings(conn, product_ids: list[int]) -> dict[int, str]:
    if not product_ids:
        return {}
    rows = conn.execute(text("""
        SELECT product_id, embedding
        FROM product_embeddings
        WHERE product_id = ANY(:pids)
    """), {"pids": product_ids}).mappings().all()
    return {r["product_id"]: r["embedding"] for r in rows}


def _find_similar_with_popularity(conn, centroid: str, exclude_ids: set[int], top_k: int) -> list[tuple[int, float]]:
    """
    Tìm sản phẩm tương tự + tính popularity score từ:
      - Số lần mua (orders)
      - Số lần thêm vào cart (user_interactions)
      - Số lần view (user_interactions)
    """
    if not exclude_ids:
        exclude_ids = {-1}

    if not centroid:
        return _get_popular_raw(conn, exclude_ids, top_k)

    rows = conn.execute(text("""
        WITH popularity AS (
            SELECT ps.product_id,
                   COALESCE(SUM(oi.quantity), 0)::float +
                   COALESCE(
                       (SELECT COUNT(*)::float FROM user_interactions ui
                        WHERE ui.product_id = ps.product_id
                        AND ui.interaction_type = 'ADD_TO_CART'), 0) AS pop_score
            FROM product_skus ps
            LEFT JOIN order_items oi ON oi.product_sku_id = ps.id
            LEFT JOIN orders o ON o.id = oi.order_id AND o.order_status != 'CANCELLED'
            GROUP BY ps.product_id
        )
        SELECT DISTINCT
            pe.product_id,
            MIN(pe.embedding <=> CAST(:vector AS vector)) AS distance,
            COALESCE(p2.pop_score, 0) AS popularity
        FROM product_embeddings pe
        JOIN products p ON p.id = pe.product_id
        LEFT JOIN popularity p2 ON p2.product_id = pe.product_id
        WHERE p.is_active = TRUE
          AND pe.product_id != ALL(:exclude_ids)
        GROUP BY pe.product_id, p2.pop_score
        ORDER BY distance ASC, popularity DESC
        LIMIT :top_k
    """), {
        "vector": centroid,
        "exclude_ids": list(exclude_ids),
        "top_k": top_k,
    }).mappings().all()

    return [(r["product_id"], float(r["distance"])) for r in rows]


def _get_popular_raw(conn, exclude_ids: set[int], limit: int) -> list[tuple[int, float]]:
    """Fallback: sản phẩm phổ biến nhất (không có centroid)."""
    rows = conn.execute(text("""
        SELECT ps.product_id, SUM(oi.quantity) AS total_sold
        FROM product_skus ps
        LEFT JOIN order_items oi ON oi.product_sku_id = ps.id
        LEFT JOIN orders o ON o.id = oi.order_id AND o.order_status != 'CANCELLED'
        JOIN products p ON p.id = ps.product_id
        WHERE p.is_active = TRUE
          AND ps.product_id != ALL(:exclude_ids)
        GROUP BY ps.product_id
        ORDER BY total_sold DESC
        LIMIT :limit
    """), {"exclude_ids": list(exclude_ids), "limit": limit}).mappings().all()
    return [(r["product_id"], float(r["total_sold"])) for r in rows]


def _get_popular_products(conn, limit: int, exclude_ids: set[int]) -> list[int]:
    """Fallback: sản phẩm phổ biến cho user mới (loại trừ đã mua/wishlist)."""
    if not exclude_ids:
        exclude_ids = {-1}

    rows = conn.execute(text("""
        SELECT ps.product_id, SUM(oi.quantity) AS total_sold
        FROM product_skus ps
        LEFT JOIN order_items oi ON oi.product_sku_id = ps.id
        LEFT JOIN orders o ON o.id = oi.order_id AND o.order_status != 'CANCELLED'
        JOIN products p ON p.id = ps.product_id
        WHERE p.is_active = TRUE
          AND ps.product_id != ALL(:exclude_ids)
        GROUP BY ps.product_id
        ORDER BY total_sold DESC
        LIMIT :limit
    """), {"exclude_ids": list(exclude_ids), "limit": limit}).mappings().all()
    return [r["product_id"] for r in rows]
