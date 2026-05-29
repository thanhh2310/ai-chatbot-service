import logging
from sqlalchemy import text
from app.services.db_service import db
import numpy as np
import math
from datetime import datetime, timezone

engine = db.get_engine()
logger = logging.getLogger(__name__)

_SEEDS_PER_SOURCE = 3
_MAX_CANDIDATES = 80


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
            # ── 1. Sản phẩm loại trừ (đã mua thành công) ─────────────────────
            exclude_ids = _get_excluded_ids(conn, user_id)

            # ── 2. Hồ sơ user + seed products đa tín hiệu ─────────────────────
            user_profile = _get_user_profile(conn, user_id)
            brand_reliable = _is_brand_reliable(conn)
            seed_scores = _get_seed_scores(conn, user_id)
            seed_ids = list(seed_scores.keys())

            # ── 3. Fallback: user mới → sản phẩm phổ biến ──────────────────────
            if not seed_ids:
                return _get_cold_start_products(conn, limit, exclude_ids, user_profile)

            # ── 4. Embeddings của seed products ────────────────────────────────
            seed_vectors = _get_embeddings(conn, seed_ids)
            if not seed_vectors:
                return _get_cold_start_products(conn, limit, exclude_ids, user_profile)

            # ── 5. Tính centroid (vector trung bình có trọng số) ─────────────────
            centroid = _compute_weighted_centroid(seed_scores, seed_ids, seed_vectors)

            # ── 6. Tìm ứng viên và rerank bằng hành vi + review + profile ─────
            candidates = _find_hybrid_candidates(conn, centroid, exclude_ids, _MAX_CANDIDATES)
            reranked = _rerank_candidates(conn, user_id, candidates, seed_scores, user_profile, limit, brand_reliable)

            logger.info(f"✅ Recommendations user={user_id}: {[pid for pid, _ in reranked]}")
            return [pid for pid, _ in reranked]

    except Exception as e:
        logger.error(f"❌ Recommendation lỗi user={user_id}: {e}", exc_info=True)
        return []


# ── PRIVATE HELPERS ──────────────────────────────────────────────────────────

def _get_excluded_ids(conn, user_id: int) -> set[int]:
    """Loại trừ sản phẩm đã mua thành công; wishlist vẫn là tín hiệu sở thích."""
    rows = conn.execute(text("""
        SELECT DISTINCT ps.product_id
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN product_skus ps ON oi.product_sku_id = ps.id
        WHERE o.user_id = :uid
          AND o.order_status IN ('PROCESSING', 'SHIPPED', 'DELIVERED')
    """), {"uid": user_id}).scalars().all()
    return set(rows)


def _get_user_profile(conn, user_id: int) -> dict:
    row = conn.execute(text("""
        SELECT id, height, weight
        FROM users
        WHERE id = :uid AND deleted_at IS NULL
    """), {"uid": user_id}).mappings().first()
    if not row:
        return {}
    height = float(row["height"]) if row["height"] is not None else None
    weight = float(row["weight"]) if row["weight"] is not None else None
    return {
        "height": height,
        "weight": weight,
        "bmi": _compute_bmi(height, weight),
        "likely_sizes": _estimate_clothing_sizes(height, weight),
    }


def _is_brand_reliable(conn) -> bool:
    count = conn.execute(text("""
        SELECT COUNT(DISTINCT brand_id) FILTER (WHERE brand_id IS NOT NULL)
        FROM products
        WHERE is_active = TRUE
          AND EXISTS (
              SELECT 1
              FROM product_skus sk
              WHERE sk.product_id = products.id
                AND sk.is_active = TRUE
                AND sk.stock_quantity > 0
          )
    """)).scalar()
    return int(count or 0) > 1


def _get_seed_scores(conn, user_id: int) -> dict[int, float]:
    """
    Lấy seed products từ order_status_history, orders, reviews, interactions, wishlist.
    PURCHASE > CART > WISHLIST > VIEW và có time-decay theo độ mới.
    """
    rows = conn.execute(text("""
        WITH seeds AS (
            SELECT
                ps.product_id,
                CASE
                    WHEN o.order_status = 'DELIVERED' THEN 5.0
                    WHEN o.order_status IN ('PROCESSING', 'SHIPPED') THEN 4.0
                    WHEN o.order_status = 'CANCELLED' THEN -4.0
                    ELSE 2.0
                END * GREATEST(1, oi.quantity) AS base_weight,
                o.created_at
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN product_skus ps ON oi.product_sku_id = ps.id
            WHERE o.user_id = :uid

            UNION ALL

            SELECT product_id, 2.5 AS base_weight, created_at
            FROM wishlists
            WHERE user_id = :uid

            UNION ALL

            SELECT
                ui.product_id,
                CASE ui.interaction_type
                    WHEN 'PURCHASE' THEN 5.0
                    WHEN 'ADD_TO_CART' THEN 3.0
                    WHEN 'VIEW' THEN 1.0
                    WHEN 'SEARCH' THEN 0.5
                    ELSE COALESCE(ui.interaction_weight, 1.0)
                END AS base_weight,
                ui.created_at
            FROM user_interactions ui
            WHERE ui.user_id = :uid AND ui.product_id IS NOT NULL

            UNION ALL

            SELECT
                r.product_id,
                CASE
                    WHEN r.rating >= 4 THEN 1.5 + ((r.rating - 4) * 0.5)
                    WHEN r.rating <= 2 THEN -2.0
                    ELSE 0.2
                END AS base_weight,
                r.created_at
            FROM reviews r
            WHERE r.user_id = :uid

            UNION ALL

            SELECT
                ps.product_id,
                CASE
                    WHEN osh.status IN ('CANCELLED', 'REFUNDED', 'RETURNED') THEN -3.5
                    WHEN osh.status = 'DELIVERED' THEN 1.0
                    ELSE 0.0
                END AS base_weight,
                osh.created_at
            FROM order_status_history osh
            JOIN orders o ON o.id = osh.order_id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN product_skus ps ON ps.id = oi.product_sku_id
            WHERE o.user_id = :uid
        )
        SELECT
            product_id,
            SUM(base_weight * EXP(-EXTRACT(EPOCH FROM (NOW() - created_at)) / (86400.0 * 45.0))) AS total_weight
        FROM seeds
        WHERE product_id IS NOT NULL
        GROUP BY product_id
        HAVING SUM(base_weight * EXP(-EXTRACT(EPOCH FROM (NOW() - created_at)) / (86400.0 * 45.0))) > 0
        ORDER BY total_weight DESC
        LIMIT :limit
    """), {"uid": user_id, "limit": _SEEDS_PER_SOURCE * 6}).mappings().all()
    return {r["product_id"]: float(r["total_weight"]) for r in rows}


def _compute_weighted_centroid(seed_scores: dict[int, float], seed_ids: list[int], seed_vectors: dict) -> str:
    """
    Tính centroid có trọng số dựa trên interaction_weight và tần suất xuất hiện.
    """
    total_weight = sum(seed_scores.values()) if seed_scores else 1.0

    vector_arrays = []
    for pid in seed_ids:
        if pid not in seed_vectors:
            continue
        w = seed_scores.get(pid, 1.0) / total_weight
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


def _find_hybrid_candidates(conn, centroid: str, exclude_ids: set[int], top_k: int) -> list[dict]:
    """
    Tìm sản phẩm tương tự + tính popularity score từ:
      - Số lần mua (orders)
      - Số lần thêm vào cart (user_interactions)
      - Số lần view (user_interactions)
    """
    if not exclude_ids:
        exclude_ids = {-1}

    if not centroid:
        return [{"product_id": pid, "distance": 1.0, "content": ""} for pid, _ in _get_popular_raw(conn, exclude_ids, top_k)]

    rows = conn.execute(text("""
        SELECT DISTINCT
            pe.product_id,
            MIN(pe.embedding <=> CAST(:vector AS vector)) AS distance,
            MAX(pe.content) AS content
        FROM product_embeddings pe
        JOIN products p ON p.id = pe.product_id
        WHERE p.is_active = TRUE
          AND pe.product_id != ALL(:exclude_ids)
          AND EXISTS (
              SELECT 1
              FROM product_skus sk
              WHERE sk.product_id = p.id
                AND sk.is_active = TRUE
                AND sk.stock_quantity > 0
          )
        GROUP BY pe.product_id
        ORDER BY distance ASC
        LIMIT :top_k
    """), {
        "vector": centroid,
        "exclude_ids": list(exclude_ids),
        "top_k": top_k,
    }).mappings().all()

    return [dict(r) for r in rows]


def _rerank_candidates(
    conn,
    user_id: int,
    candidates: list[dict],
    seed_scores: dict[int, float],
    user_profile: dict,
    limit: int,
    brand_reliable: bool = True,
) -> list[tuple[int, float]]:
    if not candidates:
        return []

    product_ids = [r["product_id"] for r in candidates]
    stats = _get_candidate_stats(conn, user_id, product_ids)
    max_seed = max(seed_scores.values()) if seed_scores else 1.0
    max_pop = max((s["popularity"] for s in stats.values()), default=1.0) or 1.0

    scored = []
    for row in candidates:
        pid = row["product_id"]
        s = stats.get(pid, {})
        vector_score = max(0.0, 1.0 - (float(row["distance"]) / 2.0))
        behavior_score = min(seed_scores.get(pid, 0.0) / max_seed, 1.0)
        popularity_score = min(float(s.get("popularity", 0.0)) / max_pop, 1.0)
        review_score = _normalize_rating(s.get("avg_rating"))
        recency_score = _normalize_recency(s.get("last_event_at"))
        body_fit_score = _body_fit_score(row.get("content") or "", user_profile, s.get("category_name"))
        penalty = float(s.get("negative_penalty", 0.0))

        if brand_reliable:
            final_score = (
                0.36 * vector_score
                + 0.20 * behavior_score
                + 0.14 * review_score
                + 0.10 * recency_score
                + 0.08 * popularity_score
                + 0.12 * body_fit_score
                - penalty
            )
        else:
            final_score = (
                0.28 * vector_score
                + 0.24 * behavior_score
                + 0.16 * review_score
                + 0.12 * recency_score
                + 0.08 * popularity_score
                + 0.12 * body_fit_score
                - penalty
            )
        scored.append((pid, final_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return _dedupe_ranked(scored, limit)


def _get_candidate_stats(conn, user_id: int, product_ids: list[int]) -> dict[int, dict]:
    rows = conn.execute(text("""
        WITH order_stats AS (
            SELECT
                ps.product_id,
                SUM(CASE WHEN o.order_status != 'CANCELLED' THEN oi.quantity ELSE 0 END)::float AS purchase_qty,
                MAX(o.created_at) AS last_order_at,
                SUM(CASE WHEN o.order_status = 'CANCELLED' OR o.payment_status = 'REFUNDED' THEN 1 ELSE 0 END)::float AS bad_orders
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN product_skus ps ON ps.id = oi.product_sku_id
            WHERE ps.product_id = ANY(:pids)
            GROUP BY ps.product_id
        ),
        interaction_stats AS (
            SELECT
                product_id,
                SUM(CASE interaction_type
                    WHEN 'PURCHASE' THEN 5.0
                    WHEN 'ADD_TO_CART' THEN 3.0
                    WHEN 'VIEW' THEN 1.0
                    WHEN 'SEARCH' THEN 0.5
                    ELSE COALESCE(interaction_weight, 1.0)
                END)::float AS interaction_score,
                MAX(created_at) AS last_interaction_at
            FROM user_interactions
            WHERE product_id = ANY(:pids)
            GROUP BY product_id
        ),
        review_stats AS (
            SELECT
                product_id,
                AVG(rating)::float AS avg_rating,
                COUNT(*) FILTER (WHERE rating >= 4)::float AS positive_reviews,
                COUNT(*) FILTER (WHERE rating <= 2)::float AS negative_reviews
            FROM reviews
            WHERE product_id = ANY(:pids) AND is_approved = TRUE
            GROUP BY product_id
        ),
        user_negative AS (
            SELECT
                ps.product_id,
                SUM(CASE WHEN osh.status IN ('CANCELLED', 'REFUNDED', 'RETURNED') THEN 1 ELSE 0 END)::float AS status_penalty
            FROM order_status_history osh
            JOIN orders o ON o.id = osh.order_id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN product_skus ps ON ps.id = oi.product_sku_id
            WHERE o.user_id = :uid AND ps.product_id = ANY(:pids)
            GROUP BY ps.product_id
        )
        SELECT
            p.id AS product_id,
            c.name AS category_name,
            COALESCE(os.purchase_qty, 0) + COALESCE(ist.interaction_score, 0) AS popularity,
            GREATEST(
                COALESCE(os.last_order_at, TIMESTAMP '1970-01-01'),
                COALESCE(ist.last_interaction_at, TIMESTAMP '1970-01-01')
            ) AS last_event_at,
            rs.avg_rating,
            LEAST(
                0.45,
                COALESCE(un.status_penalty, 0) * 0.18
                + COALESCE(os.bad_orders, 0) * 0.08
                + COALESCE(rs.negative_reviews, 0) * 0.03
            ) AS negative_penalty
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN order_stats os ON os.product_id = p.id
        LEFT JOIN interaction_stats ist ON ist.product_id = p.id
        LEFT JOIN review_stats rs ON rs.product_id = p.id
        LEFT JOIN user_negative un ON un.product_id = p.id
        WHERE p.id = ANY(:pids)
    """), {"uid": user_id, "pids": product_ids}).mappings().all()
    return {r["product_id"]: dict(r) for r in rows}


def _get_popular_raw(conn, exclude_ids: set[int], limit: int) -> list[tuple[int, float]]:
    """Fallback: sản phẩm phổ biến nhất (không có centroid)."""
    rows = conn.execute(text("""
        SELECT ps.product_id, COALESCE(SUM(oi.quantity), 0) AS total_sold
        FROM product_skus ps
        LEFT JOIN order_items oi ON oi.product_sku_id = ps.id
        LEFT JOIN orders o ON o.id = oi.order_id AND o.order_status != 'CANCELLED'
        JOIN products p ON p.id = ps.product_id
        WHERE p.is_active = TRUE
          AND ps.product_id != ALL(:exclude_ids)
          AND ps.is_active = TRUE
          AND ps.stock_quantity > 0
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
        SELECT ps.product_id, COALESCE(SUM(oi.quantity), 0) AS total_sold
        FROM product_skus ps
        LEFT JOIN order_items oi ON oi.product_sku_id = ps.id
        LEFT JOIN orders o ON o.id = oi.order_id AND o.order_status != 'CANCELLED'
        JOIN products p ON p.id = ps.product_id
        WHERE p.is_active = TRUE
          AND ps.product_id != ALL(:exclude_ids)
          AND ps.is_active = TRUE
          AND ps.stock_quantity > 0
        GROUP BY ps.product_id
        ORDER BY total_sold DESC
        LIMIT :limit
    """), {"exclude_ids": list(exclude_ids), "limit": limit}).mappings().all()
    return [r["product_id"] for r in rows]


def _get_cold_start_products(conn, limit: int, exclude_ids: set[int], user_profile: dict) -> list[int]:
    popular = _get_popular_raw(conn, exclude_ids or {-1}, max(limit * 3, limit))
    if not user_profile:
        return [pid for pid, _ in popular[:limit]]

    rows = []
    if popular:
        rows = conn.execute(text("""
            SELECT pe.product_id, pe.content, c.name AS category_name
            FROM product_embeddings pe
            JOIN products p ON p.id = pe.product_id
            LEFT JOIN categories c ON c.id = p.category_id
            WHERE pe.product_id = ANY(:pids)
        """), {"pids": [pid for pid, _ in popular]}).mappings().all()
    by_id = {r["product_id"]: r for r in rows}
    scored = []
    for pid, pop_score in popular:
        row = by_id.get(pid, {})
        score = float(pop_score or 0.0) + _body_fit_score(row.get("content") or "", user_profile, row.get("category_name")) * 2.0
        scored.append((pid, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [pid for pid, _ in scored[:limit]]


def _compute_bmi(height_cm: float | None, weight_kg: float | None) -> float | None:
    if not height_cm or not weight_kg or height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)


def _estimate_clothing_sizes(height_cm: float | None, weight_kg: float | None) -> list[str]:
    if not height_cm or not weight_kg:
        return []
    bmi = _compute_bmi(height_cm, weight_kg) or 0
    if height_cm < 160 and weight_kg < 55:
        return ["S", "M"]
    if height_cm < 170 and weight_kg < 68:
        return ["M", "S", "L"]
    if height_cm < 180 and weight_kg < 82:
        return ["L", "M", "XL"]
    if bmi >= 27 or weight_kg >= 82:
        return ["XL", "XXL", "L"]
    return ["M", "L"]


def _normalize_rating(avg_rating) -> float:
    if avg_rating is None:
        return 0.5
    return max(0.0, min((float(avg_rating) - 1.0) / 4.0, 1.0))


def _normalize_recency(last_event_at) -> float:
    if not last_event_at:
        return 0.0
    try:
        now = datetime.now(last_event_at.tzinfo or timezone.utc).replace(tzinfo=last_event_at.tzinfo)
        days = max((now - last_event_at).days, 0)
        return math.exp(-days / 45.0)
    except Exception:
        return 0.0


def _body_fit_score(content: str, user_profile: dict, category_name: str | None) -> float:
    if not user_profile:
        return 0.0
    category = (category_name or "").lower()
    content_lower = content.lower()
    if not any(token in category or token in content_lower for token in ["quần", "áo", "clothes", "apparel"]):
        return 0.0

    score = 0.15
    likely_sizes = [size.lower() for size in user_profile.get("likely_sizes", [])]
    if likely_sizes and any(f" {size} " in f" {content_lower} " or f"size {size}" in content_lower for size in likely_sizes):
        score += 0.55

    bmi = user_profile.get("bmi")
    if bmi:
        if bmi >= 25 and any(token in content_lower for token in ["co giãn", "thoải mái", "regular", "relaxed", "stretch"]):
            score += 0.20
        elif bmi < 20 and any(token in content_lower for token in ["slim", "ôm", "fit", "compression"]):
            score += 0.15
    return min(score, 1.0)


def _dedupe_ranked(scored: list[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
    seen = set()
    result = []
    for pid, score in scored:
        if pid in seen:
            continue
        seen.add(pid)
        result.append((pid, score))
        if len(result) >= limit:
            break
    return result
