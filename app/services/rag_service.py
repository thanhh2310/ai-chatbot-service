import logging
import re  # BẮT BUỘC THÊM IMPORT NÀY
from sqlalchemy import text
from app.services.db_service import db
from app.services.intent_service import analyze_query
from app.services.embedding_service import embed_query

logger = logging.getLogger(__name__)

# Dung sai ngân sách: +15% để không bỏ sót sản phẩm cận giá
_BUDGET_TOLERANCE = 1.15

# Số ứng viên lấy ở giai đoạn 1 (luôn gấp bội top_k để có đủ để rerank)
_CANDIDATE_MULTIPLIER = 4


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


def _build_where_clauses(intent: dict, target_category_id) -> tuple[list[str], dict]:
    """
    Trả về (where_clauses, params) cho bộ lọc cứng ở giai đoạn 1.
    Chỉ áp dụng các filter có độ tin cậy cao (category, brand, budget).
    """
    clauses: list[str] = ["p.is_active = TRUE"]
    params: dict = {}

    # Lọc danh mục — ưu tiên từ UI (caller), fallback sang AI
    if cat_id := target_category_id or intent.get("category_id"):
        clauses.append("pe.category_id = :cat_id")
        params["cat_id"] = cat_id

    # Lọc thương hiệu
    if brand := intent.get("brand_name"):
        clauses.append("b.name ILIKE :brand")
        params["brand"] = f"%{brand}%"

    # Lọc ngân sách (+15% dung sai)
    if budget := intent.get("max_budget"):
        clauses.append("p.base_price <= :max_budget")
        params["max_budget"] = int(budget * _BUDGET_TOLERANCE)

    return clauses, params


def _rerank(candidates: list[dict], intent: dict, top_k: int) -> list[int]:
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
        vector_score = 1.0 - (float(row["distance"]) / 2.0)

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
        boost += matched * 0.05

        scored.append((row["product_id"], vector_score + boost))

    # Sắp xếp giảm dần theo final score (điểm càng cao càng giống)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:top_k]]


def search_similar_products(
    query_text: str,
    target_category_id: int | None = None,
    top_k: int = 5,
) -> list[int]:
    session = db.get_session()
    try:
        # ── Giai đoạn 0: Phân tích ý định ──────────────────────────────
        intent = analyze_query(query_text)

        # ── Giai đoạn 1: Vector search — lấy nhiều ứng viên (top_k * 4) ─
        enhanced_query = _build_enhanced_query(intent)
        query_vector   = embed_query(enhanced_query)

        where_clauses, params = _build_where_clauses(intent, target_category_id)
        where_sql = "WHERE " + " AND ".join(where_clauses)

        # Lấy gấp bội để giai đoạn 2 có đủ ứng viên sau khi lọc excluded
        candidate_limit = top_k * _CANDIDATE_MULTIPLIER
        params.update({"vector": str(query_vector), "limit": candidate_limit})

        sql = text(f"""
            SELECT
                pe.product_id,
                pe.content,
                (pe.embedding <=> CAST(:vector AS vector)) AS distance
            FROM product_embeddings pe
            JOIN products p  ON p.id  = pe.product_id
            LEFT JOIN brands b ON p.brand_id = b.id
            {where_sql}
            ORDER BY distance ASC
            LIMIT :limit
        """)

        rows = session.execute(sql, params).mappings().all()

        # ── Giai đoạn 2: Rerank + lọc excluded ─────────────────────────
        product_ids = _rerank(list(rows), intent, top_k)

        logger.info(
            f"✅ Query='{query_text}' | Intent={intent} | "
            f"Candidates={len(rows)} → Final={len(product_ids)}"
        )
        return product_ids

    except Exception as e:
        logger.error(f"❌ RAG Service lỗi: {e}", exc_info=True)
        return []
    finally:
        session.close()


