import logging
import re  # BẮT BUỘC THÊM IMPORT NÀY
from sqlalchemy import text
from app.services.db_service import db
from app.services.intent_service import analyze_query
from app.services.embedding_service import embed_query
from app.utils.config import Config
from app.utils.ttl_cache import TTLCache

logger = logging.getLogger(__name__)

# Dung sai ngân sách: +15% để không bỏ sót sản phẩm cận giá
_BUDGET_TOLERANCE = 1.15

# Số ứng viên lấy ở giai đoạn 1. Lấy rộng hơn để rerank có đủ ứng viên khi vector miss.
_CANDIDATE_MULTIPLIER = 12
_MIN_CANDIDATES = 40
_SEARCH_CACHE = TTLCache(ttl_seconds=Config.AI_CACHE_TTL_SECONDS, max_size=1024)

_BRAND_ALIASES = {
    "nike": ["nike", "nai"],
    "adidas": ["adidas", "das"],
    "under armour": ["under armour", "ua"],
    "puma": ["puma"],
    "li-ning": ["li-ning", "lining"],
    "new balance": ["new balance", "nb"],
}


def _build_enhanced_query(intent: dict) -> str:
    """
    Xây dựng chuỗi query phong phú hơn để embedding vector chính xác hơn.
    Thêm context về màu sắc tích cực, giới tính, môn thể thao.
    """
    parts = [intent.get("search_keywords") or ""]

    if (gender := intent.get("gender")) and gender != "unisex":
        parts.append(f"dành cho {gender}")
    if sport := intent.get("sport_type"):
        parts.append(f"dùng cho {sport}")
        if sport == "gym":
            parts.append("thoáng mát thấm hút mồ hôi co giãn")
        elif sport == "chạy bộ":
            parts.append("nhẹ êm ái đàn hồi tốt")
        elif sport == "bóng đá":
            parts.append("thi đấu bám sân")
            
    if color := intent.get("color_preference"):
        parts.append(f"màu {color}")
    if tier := intent.get("price_tier"):
        tier_map = {"binh_dan": "giá rẻ phổ thông", "tam_trung": "tầm trung", "cao_cap": "cao cấp chuyên nghiệp"}
        parts.append(tier_map.get(tier, ""))

    return " ".join(filter(None, parts))


def _prepare_intent_for_catalog(session, intent: dict) -> dict:
    prepared = dict(intent)
    brand = prepared.get("brand_name")
    if brand and not _brand_filter_applicable(session, brand):
        prepared["brand_name"] = None
        prepared["search_keywords"] = _strip_brand_terms(prepared.get("search_keywords") or "", brand)
        logger.info("Brand filter ignored because catalog brand data is missing, sparse, or single-brand: %s", brand)
    return prepared


def _brand_filter_applicable(session, brand: str | None) -> bool:
    if not brand:
        return False
    row = session.execute(text("""
        SELECT
            COUNT(DISTINCT b.id) FILTER (WHERE b.id IS NOT NULL) AS brand_count,
            COUNT(*) FILTER (WHERE b.name ILIKE :brand) AS matching_products
        FROM products p
        LEFT JOIN brands b ON b.id = p.brand_id
        WHERE p.is_active = TRUE
          AND EXISTS (
              SELECT 1
              FROM product_skus sk
              WHERE sk.product_id = p.id
                AND sk.is_active = TRUE
                AND sk.stock_quantity > 0
          )
    """), {"brand": f"%{brand}%"}).mappings().first()
    if not row:
        return False
    return int(row["brand_count"] or 0) > 1 and int(row["matching_products"] or 0) > 0


def _strip_brand_terms(query: str, brand: str) -> str:
    terms = _BRAND_ALIASES.get(brand.lower(), [brand])
    clean = query
    for term in terms:
        clean = re.sub(rf'(?:^|\s|\W|_){re.escape(term)}(?:$|\s|\W|_)', ' ', clean, flags=re.IGNORECASE | re.UNICODE)
    return re.sub(r"\s+", " ", clean).strip()


def _keyword_terms(intent: dict) -> list[str]:
    raw = " ".join(filter(None, [
        intent.get("search_keywords") or "",
        intent.get("sport_type") or "",
        intent.get("color_preference") or "",
        intent.get("price_tier") or "",
    ])).lower()
    stopwords = {
        "cho", "cua", "của", "toi", "tôi", "anh", "chị", "em", "can", "cần",
        "muon", "muốn", "mua", "san", "sản", "pham", "phẩm", "hang", "hàng",
        "gia", "giá", "voi", "với", "khong", "không", "lay", "lấy",
    }
    terms = []
    for term in re.findall(r"[\wÀ-ỹ-]+", raw, flags=re.UNICODE):
        if len(term) < 2 or term in stopwords:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:10]


def _build_where_clauses(intent: dict, target_category_id) -> tuple[list[str], dict]:
    """
    Trả về (where_clauses, params) cho bộ lọc cứng ở giai đoạn 1.
    Chỉ áp dụng các filter có độ tin cậy cao (category, brand, budget).
    """
    clauses: list[str] = ["p.is_active = TRUE"]
    params: dict = {}

    # Lọc danh mục — ưu tiên từ UI (caller), fallback sang AI
    if cat_id := target_category_id or intent.get("category_id"):
        clauses.append("p.category_id = :cat_id")
        params["cat_id"] = cat_id

    # Lọc thương hiệu
    if brand := intent.get("brand_name"):
        clauses.append("b.name ILIKE :brand")
        params["brand"] = f"%{brand}%"

    # Lọc ngân sách (+15% dung sai)
    if budget := intent.get("max_budget"):
        clauses.append("p.base_price <= :max_budget")
        params["max_budget"] = int(budget * _BUDGET_TOLERANCE)

    clauses.append("""
        EXISTS (
            SELECT 1
            FROM product_skus sk
            WHERE sk.product_id = p.id
              AND sk.is_active = TRUE
              AND sk.stock_quantity > 0
        )
    """)

    return clauses, params


def _merge_candidates(*candidate_lists: list[dict]) -> list[dict]:
    merged: dict[int, dict] = {}
    for candidates in candidate_lists:
        for row in candidates:
            pid = row["product_id"]
            existing = merged.get(pid)
            if not existing:
                merged[pid] = dict(row)
                continue
            existing["distance"] = min(float(existing.get("distance", 1.2)), float(row.get("distance", 1.2)))
            existing["lexical_score"] = max(float(existing.get("lexical_score", 0.0)), float(row.get("lexical_score", 0.0)))
            if len(row.get("content") or "") > len(existing.get("content") or ""):
                existing["content"] = row.get("content")
    return list(merged.values())


def _find_vector_candidates(session, query_vector: list[float], intent: dict, target_category_id: int | None, limit: int) -> list[dict]:
    where_clauses, params = _build_where_clauses(intent, target_category_id)
    where_sql = "WHERE " + " AND ".join(where_clauses)
    params.update({"vector": str(query_vector), "limit": limit})

    sql = text(f"""
        SELECT
            pe.product_id,
            CONCAT(pe.content, ' ', COALESCE(attrs.in_stock_attributes, '')) AS content,
            (pe.embedding <=> CAST(:vector AS vector)) AS distance,
            0.0::float AS lexical_score
        FROM product_embeddings pe
        JOIN products p  ON p.id  = pe.product_id
        LEFT JOIN brands b ON p.brand_id = b.id
        LEFT JOIN categories c ON c.id = p.category_id
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
        {where_sql}
        ORDER BY distance ASC
        LIMIT :limit
    """)
    return [dict(r) for r in session.execute(sql, params).mappings().all()]


def _find_lexical_candidates(session, intent: dict, target_category_id: int | None, limit: int) -> list[dict]:
    terms = _keyword_terms(intent)
    if not terms:
        return []

    where_clauses, params = _build_where_clauses(intent, target_category_id)
    text_blob = "LOWER(CONCAT_WS(' ', p.name, p.description, b.name, c.name, attrs.in_stock_attributes))"

    term_filters = []
    score_parts = []
    for i, term_value in enumerate(terms):
        key = f"term_{i}"
        params[key] = f"%{term_value}%"
        term_filters.append(f"{text_blob} LIKE :{key}")
        score_parts.append(f"""
            CASE
                WHEN LOWER(p.name) LIKE :{key} THEN 4.0
                WHEN LOWER(COALESCE(attrs.in_stock_attributes, '')) LIKE :{key} THEN 3.0
                WHEN LOWER(COALESCE(c.name, '')) LIKE :{key} THEN 2.0
                WHEN LOWER(COALESCE(p.description, '')) LIKE :{key} THEN 1.0
                ELSE 0.0
            END
        """)

    where_clauses.append("(" + " OR ".join(term_filters) + ")")
    where_sql = "WHERE " + " AND ".join(where_clauses)
    params["limit"] = limit

    sql = text(f"""
        SELECT
            p.id AS product_id,
            CONCAT_WS(' ', p.name, p.description, b.name, c.name, attrs.in_stock_attributes) AS content,
            1.2::float AS distance,
            ({" + ".join(score_parts)}) AS lexical_score
        FROM products p
        LEFT JOIN product_embeddings pe ON pe.product_id = p.id
        LEFT JOIN brands b ON p.brand_id = b.id
        LEFT JOIN categories c ON c.id = p.category_id
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
        {where_sql}
        ORDER BY lexical_score DESC, p.updated_at DESC NULLS LAST
        LIMIT :limit
    """)
    return [dict(r) for r in session.execute(sql, params).mappings().all()]


def _find_broad_candidates(session, intent: dict, target_category_id: int | None, limit: int) -> list[dict]:
    where_clauses, params = _build_where_clauses(intent, target_category_id)
    where_sql = "WHERE " + " AND ".join(where_clauses)
    params["limit"] = limit

    sql = text(f"""
        SELECT
            p.id AS product_id,
            CONCAT_WS(' ', p.name, p.description, b.name, c.name, attrs.in_stock_attributes) AS content,
            1.4::float AS distance,
            0.0::float AS lexical_score
        FROM products p
        LEFT JOIN brands b ON p.brand_id = b.id
        LEFT JOIN categories c ON c.id = p.category_id
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
        {where_sql}
        ORDER BY p.updated_at DESC NULLS LAST, p.created_at DESC NULLS LAST
        LIMIT :limit
    """)
    return [dict(r) for r in session.execute(sql, params).mappings().all()]


def _rerank(candidates: list[dict], intent: dict, top_k: int, user_id: int | None = None, session=None) -> list[int]:
    """
    Giai đoạn 2: Rerank danh sách ứng viên bằng cách kết hợp:
      - vector_score: khoảng cách cosine (scale về 0-1)
      - boost: điểm thưởng từ keyword match trong content
    """
    excluded = [w.lower() for w in intent.get("excluded_keywords", [])]
    color_pref = (intent.get("color_preference") or "").lower()
    boost_keywords = [
        kw.lower() for kw in (intent.get("search_keywords") or "").split()
        if len(kw) > 2
    ]

    personal_features = _get_personal_features(session, user_id, [r["product_id"] for r in candidates]) if user_id and session else {}
    max_personal = max((v["score"] for v in personal_features.values()), default=1.0) or 1.0
    max_lexical = max((float(r.get("lexical_score", 0.0)) for r in candidates), default=1.0) or 1.0

    scored = []
    for row in candidates:
        content_lower = (row["content"] or "").lower()

        # 1. FIX: Loại bỏ cứng dùng Regex. Thay \b bằng ranh giới từ an toàn cho tiếng Việt
        is_excluded = False
        for excl in excluded:
            if re.search(rf'(?:^|\s|\W|_){re.escape(excl)}(?:$|\s|\W|_)', content_lower, flags=re.IGNORECASE | re.UNICODE):
                is_excluded = True
                break
        
        if is_excluded:
            continue

        # 2. FIX: Chuyển Cosine Distance (0-2) -> Cosine Similarity (0-1)
        # Điểm distance của pgvector chạy từ 0 đến 2. Chia 2 để giới hạn về 1, rồi lấy 1 trừ đi.
        vector_score = max(0.0, 1.0 - (float(row.get("distance", 1.2)) / 2.0))
        lexical_score = min(float(row.get("lexical_score", 0.0)) / max_lexical, 1.0)

        # Điểm thưởng
        boost = 0.0
        
        # 3. FIX: Thưởng nếu content khớp màu ưa thích (Tăng mạnh điểm để đè bẹp Vector_score nếu sai màu)
        if color_pref and re.search(rf'(?:^|\s|\W|_){re.escape(color_pref)}(?:$|\s|\W|_)', content_lower, flags=re.IGNORECASE | re.UNICODE):
            boost += 0.25
            
        # 4. FIX: Thưởng theo số keyword khớp trong content (Regex hỗ trợ tiếng Việt)
        matched = 0
        for kw in boost_keywords:
            if re.search(rf'(?:^|\s|\W|_){re.escape(kw)}(?:$|\s|\W|_)', content_lower, flags=re.IGNORECASE | re.UNICODE):
                matched += 1
        keyword_score = min(matched / max(len(boost_keywords), 1), 1.0)
        boost += matched * 0.04

        personalization = 0.0
        feature = personal_features.get(row["product_id"])
        if feature:
            personalization = min(float(feature["score"]) / max_personal, 1.0)
            boost += _profile_fit_boost(content_lower, feature)
            boost += _review_boost(feature)
            boost -= float(feature.get("penalty", 0.0))

        final_score = (
            0.42 * vector_score
            + 0.28 * lexical_score
            + 0.15 * keyword_score
            + 0.15 * personalization
            + boost
        )
        scored.append((row["product_id"], final_score))

    # Sắp xếp giảm dần theo final score (điểm càng cao càng giống)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:top_k]]


def search_similar_products(
    query_text: str,
    target_category_id: int | None = None,
    top_k: int = 5,
    user_id: int | None = None,
) -> list[int]:
    cache_key = (
        re.sub(r"\s+", " ", (query_text or "").strip().lower()),
        int(target_category_id) if target_category_id is not None else None,
        int(top_k),
        int(user_id) if user_id is not None else None,
    )
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    session = db.get_session()
    try:
        # ── Giai đoạn 0: Phân tích ý định ──────────────────────────────
        intent = analyze_query(query_text)
        intent = _prepare_intent_for_catalog(session, intent)

        # ── Giai đoạn 1: Vector + lexical/attribute candidates ──────────
        enhanced_query = _build_enhanced_query(intent)
        candidate_limit = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)
        vector_rows = []
        try:
            query_vector = embed_query(enhanced_query)
            vector_rows = _find_vector_candidates(session, query_vector, intent, target_category_id, candidate_limit)
        except Exception as embed_error:
            logger.warning("Vector search skipped, falling back to lexical search: %s", embed_error)
        lexical_rows = _find_lexical_candidates(session, intent, target_category_id, candidate_limit)
        rows = _merge_candidates(vector_rows, lexical_rows)
        if not rows:
            rows = _find_broad_candidates(session, intent, target_category_id, candidate_limit)
        if not rows and target_category_id is None and intent.get("category_id") is not None:
            relaxed_intent = dict(intent)
            relaxed_intent["category_id"] = None
            logger.info("Retrying search without inferred category filter for query='%s'", query_text)
            relaxed_lexical = _find_lexical_candidates(session, relaxed_intent, None, candidate_limit)
            relaxed_broad = _find_broad_candidates(session, relaxed_intent, None, candidate_limit)
            rows = _merge_candidates(relaxed_lexical, relaxed_broad)

        # ── Giai đoạn 2: Rerank + lọc excluded ─────────────────────────
        product_ids = _rerank(list(rows), intent, top_k, user_id=user_id, session=session)

        logger.info(
            f"✅ Query='{query_text}' | Intent={intent} | "
            f"Vector={len(vector_rows)} Lexical={len(lexical_rows)} Candidates={len(rows)} → Final={len(product_ids)}"
        )
        _SEARCH_CACHE.set(cache_key, product_ids)
        return product_ids

    except Exception as e:
        logger.error(f"❌ RAG Service lỗi: {e}", exc_info=True)
        return []
    finally:
        session.close()


def _get_personal_features(session, user_id: int, product_ids: list[int]) -> dict[int, dict]:
    if not product_ids:
        return {}
    rows = session.execute(text("""
        WITH brand_catalog AS (
            SELECT COUNT(DISTINCT brand_id) FILTER (WHERE brand_id IS NOT NULL) > 1 AS brand_reliable
            FROM products
            WHERE is_active = TRUE
        ),
        user_profile AS (
            SELECT height, weight
            FROM users
            WHERE id = :uid
        ),
        direct AS (
            SELECT
                product_id,
                SUM(CASE interaction_type
                    WHEN 'PURCHASE' THEN 5.0
                    WHEN 'ADD_TO_CART' THEN 3.0
                    WHEN 'VIEW' THEN 1.0
                    WHEN 'SEARCH' THEN 0.5
                    ELSE COALESCE(interaction_weight, 1.0)
                END * EXP(-EXTRACT(EPOCH FROM (NOW() - created_at)) / (86400.0 * 45.0))) AS score,
                MAX(created_at) AS last_seen_at
            FROM user_interactions
            WHERE user_id = :uid AND product_id = ANY(:pids)
            GROUP BY product_id
        ),
        affinity AS (
            SELECT
                p2.id AS product_id,
                SUM(CASE ui.interaction_type
                    WHEN 'PURCHASE' THEN 5.0
                    WHEN 'ADD_TO_CART' THEN 3.0
                    WHEN 'VIEW' THEN 1.0
                    WHEN 'SEARCH' THEN 0.5
                    ELSE COALESCE(ui.interaction_weight, 1.0)
                END * EXP(-EXTRACT(EPOCH FROM (NOW() - ui.created_at)) / (86400.0 * 45.0))) AS score
            FROM user_interactions ui
            JOIN products p1 ON p1.id = ui.product_id
            CROSS JOIN brand_catalog bc
            JOIN products p2 ON p2.category_id = p1.category_id
                OR (bc.brand_reliable AND p1.brand_id IS NOT NULL AND p2.brand_id = p1.brand_id)
            WHERE ui.user_id = :uid
              AND ui.product_id IS NOT NULL
              AND p2.id = ANY(:pids)
            GROUP BY p2.id
        ),
        wishlist AS (
            SELECT p2.id AS product_id, COUNT(*) * 2.5 AS score
            FROM wishlists w
            JOIN products p1 ON p1.id = w.product_id
            CROSS JOIN brand_catalog bc
            JOIN products p2 ON p2.category_id = p1.category_id
                OR (bc.brand_reliable AND p1.brand_id IS NOT NULL AND p2.brand_id = p1.brand_id)
            WHERE w.user_id = :uid AND p2.id = ANY(:pids)
            GROUP BY p2.id
        ),
        reviews AS (
            SELECT
                p2.id AS product_id,
                AVG(r.rating)::float AS user_related_rating
            FROM reviews r
            JOIN products p1 ON p1.id = r.product_id
            CROSS JOIN brand_catalog bc
            JOIN products p2 ON p2.category_id = p1.category_id
                OR (bc.brand_reliable AND p1.brand_id IS NOT NULL AND p2.brand_id = p1.brand_id)
            WHERE r.user_id = :uid AND p2.id = ANY(:pids)
            GROUP BY p2.id
        ),
        penalties AS (
            SELECT
                ps.product_id,
                SUM(CASE WHEN osh.status IN ('CANCELLED', 'REFUNDED', 'RETURNED') THEN 1 ELSE 0 END) * 0.18 AS penalty
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
            up.height,
            up.weight,
            COALESCE(d.score, 0) + COALESCE(a.score, 0) + COALESCE(w.score, 0) AS score,
            r.user_related_rating,
            COALESCE(pn.penalty, 0) AS penalty
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        CROSS JOIN user_profile up
        LEFT JOIN direct d ON d.product_id = p.id
        LEFT JOIN affinity a ON a.product_id = p.id
        LEFT JOIN wishlist w ON w.product_id = p.id
        LEFT JOIN reviews r ON r.product_id = p.id
        LEFT JOIN penalties pn ON pn.product_id = p.id
        WHERE p.id = ANY(:pids)
    """), {"uid": user_id, "pids": product_ids}).mappings().all()
    return {r["product_id"]: dict(r) for r in rows}


def _review_boost(feature: dict) -> float:
    rating = feature.get("user_related_rating")
    if rating is None:
        return 0.0
    rating = float(rating)
    if rating >= 4:
        return 0.12
    if rating <= 2:
        return -0.16
    return 0.0


def _profile_fit_boost(content_lower: str, feature: dict) -> float:
    category = (feature.get("category_name") or "").lower()
    if not any(token in category or token in content_lower for token in ["quần", "áo", "clothes", "apparel"]):
        return 0.0
    height = feature.get("height")
    weight = feature.get("weight")
    if height is None or weight is None:
        return 0.0
    sizes = _estimate_sizes(float(height), float(weight))
    if any(f" {size.lower()} " in f" {content_lower} " or f"size {size.lower()}" in content_lower for size in sizes):
        return 0.18
    if any(token in content_lower for token in ["co giãn", "thoải mái", "stretch", "regular"]):
        return 0.08
    return 0.03


def _estimate_sizes(height_cm: float, weight_kg: float) -> list[str]:
    if height_cm < 160 and weight_kg < 55:
        return ["S", "M"]
    if height_cm < 170 and weight_kg < 68:
        return ["M", "S", "L"]
    if height_cm < 180 and weight_kg < 82:
        return ["L", "M", "XL"]
    if weight_kg >= 82:
        return ["XL", "XXL", "L"]
    return ["M", "L"]
