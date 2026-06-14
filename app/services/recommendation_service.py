import logging
from sqlalchemy import text
from app.services.db_service import db
import numpy as np
import math
from datetime import datetime, timezone
from app.utils.config import Config
from app.utils.ttl_cache import TTLCache

engine = db.get_engine()
logger = logging.getLogger(__name__)

_SEEDS_PER_SOURCE = 3
_MAX_CANDIDATES = 80
_RECOMMENDATION_CACHE = TTLCache(ttl_seconds=Config.AI_CACHE_TTL_SECONDS, max_size=512)


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
    cache_key = (int(user_id) if user_id is not None else None, int(limit))
    cached = _RECOMMENDATION_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

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
                product_ids = _get_cold_start_products(conn, limit, exclude_ids, user_profile)
                _RECOMMENDATION_CACHE.set(cache_key, product_ids)
                return product_ids

            # ── 4. Embeddings của seed products ────────────────────────────────
            seed_vectors = _get_embeddings(conn, seed_ids)
            if not seed_vectors:
                product_ids = _get_cold_start_products(conn, limit, exclude_ids, user_profile)
                _RECOMMENDATION_CACHE.set(cache_key, product_ids)
                return product_ids

            # ── 5. Tính centroid (vector trung bình có trọng số) ─────────────────
            centroid = _compute_weighted_centroid(seed_scores, seed_ids, seed_vectors)

            # ── 6. Tìm ứng viên và rerank bằng hành vi + review + profile ─────
            vector_candidates = _find_hybrid_candidates(conn, centroid, exclude_ids, _MAX_CANDIDATES)
            behavior_candidates = _find_behavior_candidates(conn, user_id, exclude_ids, _MAX_CANDIDATES, brand_reliable)
            broad_candidates = _find_broad_appeal_candidates(conn, exclude_ids, min(_MAX_CANDIDATES, max(limit * 4, limit)))
            candidates = _merge_candidates(vector_candidates, behavior_candidates, broad_candidates)
            reranked = _rerank_candidates(conn, user_id, candidates, seed_scores, user_profile, limit, brand_reliable)

            logger.info(f"✅ Recommendations user={user_id}: {[pid for pid, _ in reranked]}")
            product_ids = [pid for pid, _ in reranked]
            _RECOMMENDATION_CACHE.set(cache_key, product_ids)
            return product_ids

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
                ps.product_id,
                3.2 * GREATEST(1, ci.quantity) AS base_weight,
                ci.created_at
            FROM carts c
            JOIN cart_items ci ON ci.cart_id = c.id
            JOIN product_skus ps ON ps.id = ci.product_sku_id
            WHERE c.user_id = :uid

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

            UNION ALL

            SELECT
                ps.product_id,
                CASE
                    WHEN ors.status IN ('PENDING', 'APPROVED') THEN -4.0
                    WHEN ors.status = 'REJECTED' THEN -0.5
                    ELSE -2.0
                END AS base_weight,
                COALESCE(ors.updated_at, ors.created_at)
            FROM order_returns ors
            JOIN orders o ON o.id = ors.order_id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN product_skus ps ON ps.id = oi.product_sku_id
            WHERE ors.user_id = :uid
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


def _find_behavior_candidates(
    conn,
    user_id: int,
    exclude_ids: set[int],
    top_k: int,
    brand_reliable: bool,
) -> list[dict]:
    if not exclude_ids:
        exclude_ids = {-1}

    brand_join = "OR (fav.brand_id IS NOT NULL AND p.brand_id = fav.brand_id)" if brand_reliable else ""
    rows = conn.execute(text(f"""
        WITH fav AS (
            SELECT
                p.category_id,
                p.brand_id,
                SUM(signal_weight) AS score
            FROM (
                SELECT ui.product_id,
                       CASE ui.interaction_type
                           WHEN 'PURCHASE' THEN 5.0
                           WHEN 'ADD_TO_CART' THEN 3.0
                           WHEN 'VIEW' THEN 1.0
                           WHEN 'SEARCH' THEN 0.5
                           ELSE COALESCE(ui.interaction_weight, 1.0)
                       END * EXP(-EXTRACT(EPOCH FROM (NOW() - ui.created_at)) / (86400.0 * 45.0)) AS signal_weight
                FROM user_interactions ui
                WHERE ui.user_id = :uid AND ui.product_id IS NOT NULL

                UNION ALL

                SELECT w.product_id, 2.5 * EXP(-EXTRACT(EPOCH FROM (NOW() - w.created_at)) / (86400.0 * 45.0))
                FROM wishlists w
                WHERE w.user_id = :uid

                UNION ALL

                SELECT ps.product_id, 3.2 * GREATEST(1, ci.quantity) * EXP(-EXTRACT(EPOCH FROM (NOW() - ci.created_at)) / (86400.0 * 45.0))
                FROM carts c
                JOIN cart_items ci ON ci.cart_id = c.id
                JOIN product_skus ps ON ps.id = ci.product_sku_id
                WHERE c.user_id = :uid
            ) s
            JOIN products p ON p.id = s.product_id
            GROUP BY p.category_id, p.brand_id
            ORDER BY SUM(signal_weight) DESC
            LIMIT 6
        ),
        product_popularity AS (
            SELECT
                ps.product_id,
                COALESCE(SUM(oi.quantity), 0)::float
                + COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'ADD_TO_CART')::float * 0.8
                + COUNT(ui.id) FILTER (WHERE ui.interaction_type = 'VIEW')::float * 0.2 AS popularity
            FROM product_skus ps
            LEFT JOIN order_items oi ON oi.product_sku_id = ps.id
            LEFT JOIN orders o ON o.id = oi.order_id AND o.order_status != 'CANCELLED'
            LEFT JOIN user_interactions ui ON ui.product_id = ps.product_id
            GROUP BY ps.product_id
        )
        SELECT
            p.id AS product_id,
            1.0::float AS distance,
            CONCAT_WS(' ', p.name, p.description, c.name, b.name, attrs.in_stock_attributes) AS content,
            COALESCE(MAX(pp.popularity), 0) + MAX(fav.score) AS behavior_candidate_score
        FROM fav
        JOIN products p ON p.category_id = fav.category_id {brand_join}
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN brands b ON b.id = p.brand_id
        LEFT JOIN product_embeddings pe ON pe.product_id = p.id
        LEFT JOIN product_popularity pp ON pp.product_id = p.id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT a.name || ': ' || av.value, ' ' ORDER BY a.name || ': ' || av.value) AS in_stock_attributes
            FROM product_skus sk
            JOIN sku_values sv ON sv.product_sku_id = sk.id
            JOIN attribute_values av ON av.id = sv.attribute_value_id
            JOIN attributes a ON a.id = av.attribute_id
            WHERE sk.product_id = p.id
              AND sk.is_active = TRUE
              AND sk.stock_quantity > 0
        ) attrs ON TRUE
        WHERE p.is_active = TRUE
          AND p.id != ALL(:exclude_ids)
          AND EXISTS (
              SELECT 1
              FROM product_skus sk
              WHERE sk.product_id = p.id
                AND sk.is_active = TRUE
                AND sk.stock_quantity > 0
          )
        GROUP BY p.id, p.name, p.description, c.name, b.name, attrs.in_stock_attributes
        ORDER BY behavior_candidate_score DESC
        LIMIT :top_k
    """), {"uid": user_id, "exclude_ids": list(exclude_ids), "top_k": top_k}).mappings().all()
    return [dict(r) for r in rows]


def _merge_candidates(*candidate_lists: list[dict]) -> list[dict]:
    merged: dict[int, dict] = {}
    for candidates in candidate_lists:
        for row in candidates:
            pid = row["product_id"]
            existing = merged.get(pid)
            if not existing:
                merged[pid] = dict(row)
                continue
            existing["distance"] = min(float(existing.get("distance", 1.0)), float(row.get("distance", 1.0)))
            existing["behavior_candidate_score"] = max(
                float(existing.get("behavior_candidate_score", 0.0)),
                float(row.get("behavior_candidate_score", 0.0)),
            )
            if len(row.get("content") or "") > len(existing.get("content") or ""):
                existing["content"] = row.get("content")
    return list(merged.values())


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
    max_behavior_candidate = max((float(r.get("behavior_candidate_score", 0.0)) for r in candidates), default=1.0) or 1.0

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
        behavior_candidate_score = min(float(row.get("behavior_candidate_score", 0.0)) / max_behavior_candidate, 1.0)
        penalty = float(s.get("negative_penalty", 0.0))

        if brand_reliable:
            final_score = (
                0.32 * vector_score
                + 0.20 * behavior_score
                + 0.08 * behavior_candidate_score
                + 0.14 * review_score
                + 0.10 * recency_score
                + 0.04 * popularity_score
                + 0.12 * body_fit_score
                - penalty
            )
        else:
            final_score = (
                0.18 * vector_score
                + 0.24 * behavior_score
                + 0.14 * behavior_candidate_score
                + 0.16 * review_score
                + 0.12 * recency_score
                + 0.04 * popularity_score
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
        cart_stats AS (
            SELECT
                ps.product_id,
                SUM(ci.quantity)::float AS cart_qty,
                MAX(ci.created_at) AS last_cart_at
            FROM carts c
            JOIN cart_items ci ON ci.cart_id = c.id
            JOIN product_skus ps ON ps.id = ci.product_sku_id
            WHERE c.user_id = :uid AND ps.product_id = ANY(:pids)
            GROUP BY ps.product_id
        ),
        chat_stats AS (
            SELECT
                CAST(TRIM(pid) AS INTEGER) AS product_id,
                COUNT(*)::float AS chat_hits,
                MAX(cm.created_at) AS last_chat_at
            FROM chatbot_sessions cs
            JOIN chatbot_messages cm ON cm.session_id = cs.id
            CROSS JOIN LATERAL regexp_split_to_table(COALESCE(cm.retrieved_product_ids, ''), ',') AS pid
            WHERE cs.user_id = :uid
              AND TRIM(pid) ~ '^[0-9]+$'
            GROUP BY CAST(TRIM(pid) AS INTEGER)
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
        ),
        return_negative AS (
            SELECT
                ps.product_id,
                COUNT(*) FILTER (WHERE ors.status IN ('PENDING', 'APPROVED'))::float AS return_penalty
            FROM order_returns ors
            JOIN orders o ON o.id = ors.order_id
            JOIN order_items oi ON oi.order_id = o.id
            JOIN product_skus ps ON ps.id = oi.product_sku_id
            WHERE ors.user_id = :uid AND ps.product_id = ANY(:pids)
            GROUP BY ps.product_id
        )
        SELECT
            p.id AS product_id,
            c.name AS category_name,
            COALESCE(os.purchase_qty, 0)
                + COALESCE(ist.interaction_score, 0)
                + COALESCE(cs.cart_qty, 0) * 1.2
                + COALESCE(chs.chat_hits, 0) * 0.5 AS popularity,
            GREATEST(
                COALESCE(os.last_order_at, TIMESTAMP '1970-01-01'),
                COALESCE(ist.last_interaction_at, TIMESTAMP '1970-01-01'),
                COALESCE(cs.last_cart_at, TIMESTAMP '1970-01-01'),
                COALESCE(chs.last_chat_at, TIMESTAMP '1970-01-01')
            ) AS last_event_at,
            rs.avg_rating,
            LEAST(
                0.45,
                COALESCE(un.status_penalty, 0) * 0.18
                + COALESCE(rn.return_penalty, 0) * 0.25
                + COALESCE(os.bad_orders, 0) * 0.08
                + COALESCE(rs.negative_reviews, 0) * 0.03
            ) AS negative_penalty
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN order_stats os ON os.product_id = p.id
        LEFT JOIN interaction_stats ist ON ist.product_id = p.id
        LEFT JOIN cart_stats cs ON cs.product_id = p.id
        LEFT JOIN chat_stats chs ON chs.product_id = p.id
        LEFT JOIN review_stats rs ON rs.product_id = p.id
        LEFT JOIN user_negative un ON un.product_id = p.id
        LEFT JOIN return_negative rn ON rn.product_id = p.id
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
    broad_appeal = _get_broad_appeal_products(conn, exclude_ids or {-1}, max(limit * 3, limit))
    if broad_appeal:
        if not user_profile:
            return broad_appeal[:limit]

        rows = conn.execute(text("""
            SELECT pe.product_id, pe.content, c.name AS category_name
            FROM product_embeddings pe
            JOIN products p ON p.id = pe.product_id
            LEFT JOIN categories c ON c.id = p.category_id
            WHERE pe.product_id = ANY(:pids)
        """), {"pids": broad_appeal}).mappings().all()
        by_id = {r["product_id"]: r for r in rows}
        scored = []
        for index, pid in enumerate(broad_appeal):
            row = by_id.get(pid, {})
            score = (len(broad_appeal) - index) + _body_fit_score(row.get("content") or "", user_profile, row.get("category_name")) * 2.0
            scored.append((pid, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [pid for pid, _ in scored[:limit]]

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


def _find_broad_appeal_candidates(conn, exclude_ids: set[int], limit: int) -> list[dict]:
    product_ids = _get_broad_appeal_products(conn, exclude_ids or {-1}, limit)
    if not product_ids:
        return []

    rows = conn.execute(text("""
        SELECT
            p.id AS product_id,
            0.85::float AS distance,
            CONCAT_WS(' ', p.name, p.description, c.name, b.name, attrs.in_stock_attributes) AS content,
            0.0::float AS behavior_candidate_score
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN brands b ON b.id = p.brand_id
        LEFT JOIN LATERAL (
            SELECT string_agg(DISTINCT a.name || ': ' || av.value, ' ' ORDER BY a.name || ': ' || av.value) AS in_stock_attributes
            FROM product_skus sk
            JOIN sku_values sv ON sv.product_sku_id = sk.id
            JOIN attribute_values av ON av.id = sv.attribute_value_id
            JOIN attributes a ON a.id = av.attribute_id
            WHERE sk.product_id = p.id
              AND sk.is_active = TRUE
              AND sk.stock_quantity > 0
        ) attrs ON TRUE
        WHERE p.id = ANY(:preferred_ids)
        ORDER BY array_position(:preferred_ids, p.id)
    """), {"preferred_ids": product_ids}).mappings().all()
    score_by_id = {pid: float(len(product_ids) - index) for index, pid in enumerate(product_ids)}
    candidates = []
    for row in rows:
        item = dict(row)
        item["behavior_candidate_score"] = score_by_id.get(item["product_id"], 0.0)
        candidates.append(item)
    return candidates


def _get_broad_appeal_products(conn, exclude_ids: set[int], limit: int) -> list[int]:
    """Stable cold-start ranking when behavioral data is sparse or empty."""
    preferred_ids = [
        1, 2, 3, 4, 5,
        19, 20, 22, 26, 28, 29,
        31, 32, 33, 34,
        43, 44, 45, 46, 47,
        48, 49, 50, 51, 52,
        66, 67, 68, 69,
    ]
    rows = conn.execute(text("""
        SELECT p.id
        FROM products p
        WHERE p.id = ANY(:preferred_ids)
          AND p.id != ALL(:exclude_ids)
          AND p.is_active = TRUE
          AND EXISTS (
              SELECT 1
              FROM product_skus sk
              WHERE sk.product_id = p.id
                AND sk.is_active = TRUE
                AND sk.stock_quantity > 0
          )
        ORDER BY array_position(:preferred_ids, p.id)
        LIMIT :limit
    """), {
        "preferred_ids": preferred_ids,
        "exclude_ids": list(exclude_ids),
        "limit": limit,
    }).scalars().all()
    return [int(pid) for pid in rows]


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
