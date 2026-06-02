import logging
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import text
from app.services.rag_service import search_similar_products
from app.services.recommendation_service import get_recommendations_for_user
from app.services.db_service import db

engine = db.get_engine()
from app.utils.config import Config

logger = logging.getLogger(__name__)

# LLM để sinh câu trả lời
_llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    openai_api_key=Config.TOGETHER_API_KEY,
    openai_api_base="https://api.together.xyz/v1",
    temperature=0.3,
)

# Số tin nhắn lịch sử đưa vào context
_HISTORY_LIMIT = 6
# Số SP dùng làm RAG context
_RAG_TOP_K = 4

_SYSTEM_PROMPT = """
Bạn là nhân viên tư vấn bán hàng thể thao chuyên nghiệp, thân thiện.
Xưng "em", gọi khách là "anh/chị".
Chỉ tư vấn sản phẩm dựa trên thông tin được cung cấp trong phần [SẢN PHẨM] và [THÔNG TIN CÁ NHÂN HÓA].
Khi khách hỏi về size, màu hoặc thuộc tính sản phẩm, chỉ trả lời theo "Thuộc tính còn hàng" trong [SẢN PHẨM].
Với câu hỏi về sở thích, body type, size, sản phẩm thường thích, hãy dùng [THÔNG TIN CÁ NHÂN HÓA] để giải thích ngắn gọn.
Không khẳng định chắc chắn size chỉ từ chiều cao/cân nặng; hãy nói đó là ước lượng và nhắc khách kiểm tra bảng size nếu cần.
Nếu không có sản phẩm phù hợp trong ngữ cảnh, hãy thành thật nói "Em chưa tìm được sản phẩm phù hợp trong cửa hàng, anh/chị có thể mô tả rõ hơn không ạ?"
Không bịa thông tin giá, tên sản phẩm, hoặc tính năng ngoài ngữ cảnh.
Trả lời ngắn gọn, tối đa 3-4 câu.
""".strip()


def chat_with_bot(session_id: str | None, user_id: int | None, message: str) -> dict:
    """
    RAG Chatbot với multi-turn memory.

    Flow:
      1. Tạo session nếu chưa có
      2. Lưu tin nhắn user
      3. RAG: tìm SP liên quan bằng intent-aware search
      4. Kéo lịch sử dạng LangChain messages
      5. Gọi LLM với system prompt + context + history
      6. Lưu câu trả lời bot
    """
    try:
        with engine.connect() as conn:
            # ── 1. Tạo session ────────────────────────────────────────────
            if session_id:
                _ensure_session_access(conn, session_id, user_id)
            else:
                session_id = str(uuid.uuid4())
                conn.execute(text("""
                    INSERT INTO chatbot_sessions (id, user_id) VALUES (:id, :uid)
                """), {"id": session_id, "uid": user_id})
                conn.commit()

            # ── 2. Lưu tin nhắn user ──────────────────────────────────────
            conn.execute(text("""
                INSERT INTO chatbot_messages (session_id, sender_type, message_text)
                VALUES (:sid, 'USER', :msg)
            """), {"sid": session_id, "msg": message})
            conn.commit()

            # ── 3. RAG: tìm SP liên quan ──────────────────────────────────
            context_text, retrieved_ids = _retrieve_context(conn, message, user_id)
            personalization_text = _build_personalization_context(conn, user_id)

            # ── 4. Kéo lịch sử chat ───────────────────────────────────────
            history_messages = _load_history(conn, session_id, exclude_last=True)

        # ── 5. Gọi LLM ───────────────────────────────────────────────────
        bot_reply = _generate_reply(
            message=message,
            context_text=context_text,
            personalization_text=personalization_text,
            history=history_messages,
        )

        # ── 6. Lưu câu trả lời bot ───────────────────────────────────────
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO chatbot_messages
                    (session_id, sender_type, message_text, retrieved_product_ids)
                VALUES (:sid, 'BOT', :msg, :pids)
            """), {
                "sid":  session_id,
                "msg":  bot_reply,
                "pids": ",".join(map(str, retrieved_ids)),
            })
            conn.commit()

        return {
            "session_id":           session_id,
            "reply":                bot_reply,
            "suggested_product_ids": retrieved_ids,
            "product_ids":          retrieved_ids,
        }

    except Exception as e:
        logger.error(f"❌ Chatbot lỗi session={session_id}: {e}", exc_info=True)
        raise


# ── PRIVATE HELPERS ──────────────────────────────────────────────────────────

def _ensure_session_access(conn, session_id: str, user_id: int | None) -> None:
    row = conn.execute(text("""
        SELECT user_id
        FROM chatbot_sessions
        WHERE id = :sid
    """), {"sid": session_id}).mappings().first()

    if not row:
        conn.execute(text("""
            INSERT INTO chatbot_sessions (id, user_id) VALUES (:id, :uid)
        """), {"id": session_id, "uid": user_id})
        conn.commit()
        return

    owner_id = row["user_id"]
    if owner_id is not None and owner_id != user_id:
        raise PermissionError("Không được truy cập lịch sử chat của người dùng khác.")
    if owner_id is None and user_id is not None:
        conn.execute(text("""
            UPDATE chatbot_sessions
            SET user_id = :uid
            WHERE id = :sid AND user_id IS NULL
        """), {"uid": user_id, "sid": session_id})
        conn.commit()

def _retrieve_context(conn, message: str, user_id: int | None = None) -> tuple[str, list[int]]:
    """
    Sử dụng sức mạnh của Hybrid RAG thay vì Vector thuần.
    """
    # Gọi trực tiếp hàm RAG đã được tối ưu siêu việt của bạn
    # Hàm này tự động gọi Intent Service, lọc SQL giá, màu sắc và Rerank
    retrieved_ids = search_similar_products(query_text=message, top_k=_RAG_TOP_K, user_id=user_id)
    if user_id and _is_body_or_preference_question(message):
        recommendation_ids = get_recommendations_for_user(user_id=user_id, limit=_RAG_TOP_K)
        retrieved_ids = _merge_ordered_ids(retrieved_ids, recommendation_ids, _RAG_TOP_K)
    
    if not retrieved_ids:
        return "Không có sản phẩm phù hợp.", []

    # Lấy thông tin chi tiết của các ID đã lọc để nạp vào Prompt
    rows = conn.execute(text("""
        SELECT
            p.id,
            p.name,
            p.base_price,
            p.description,
            COALESCE(attrs.in_stock_attributes, '') AS in_stock_attributes
        FROM products p
        JOIN LATERAL (
            SELECT string_agg(DISTINCT a.name || ': ' || av.value, ', ' ORDER BY a.name || ': ' || av.value) AS in_stock_attributes
            FROM product_skus sk
            JOIN sku_values sv ON sv.product_sku_id = sk.id
            JOIN attribute_values av ON av.id = sv.attribute_value_id
            JOIN attributes a ON a.id = av.attribute_id
            WHERE sk.product_id = p.id
              AND sk.is_active = TRUE
              AND sk.stock_quantity > 0
        ) attrs ON TRUE
        WHERE p.id = ANY(:pids)
          AND p.is_active = TRUE
          AND EXISTS (
              SELECT 1 FROM product_skus sk
              WHERE sk.product_id = p.id
                AND sk.is_active = TRUE
                AND sk.stock_quantity > 0
          )
    """), {"pids": retrieved_ids}).mappings().all()

    context_parts = []
    available_ids = []
    rows_by_id = {r["id"]: r for r in rows}
    for pid in retrieved_ids:
        r = rows_by_id.get(pid)
        if not r:
            continue
        available_ids.append(r["id"])
        context_parts.append(
            f"- ID {r['id']}: {r['name']} | Giá: {r['base_price']} | "
            f"Thuộc tính còn hàng: {r['in_stock_attributes'] or 'chưa có'} | "
            f"Mô tả: {r['description']}"
        )
        
    context_text = "\n".join(context_parts)
    return context_text, available_ids


def _build_personalization_context(conn, user_id: int | None) -> str:
    if not user_id:
        return "Khách chưa đăng nhập; chỉ dùng nội dung câu hỏi và sản phẩm truy xuất."

    profile = conn.execute(text("""
        SELECT height, weight
        FROM users
        WHERE id = :uid
    """), {"uid": user_id}).mappings().first()

    cart_rows = conn.execute(text("""
        SELECT
            p.name,
            c.name AS category_name,
            COALESCE(b.name, '') AS brand_name,
            SUM(ci.quantity) AS quantity,
            MAX(ci.created_at) AS last_added_at
        FROM carts cart
        JOIN cart_items ci ON ci.cart_id = cart.id
        JOIN product_skus ps ON ps.id = ci.product_sku_id
        JOIN products p ON p.id = ps.product_id
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN brands b ON b.id = p.brand_id
        WHERE cart.user_id = :uid
          AND p.is_active = TRUE
          AND ps.is_active = TRUE
          AND ps.stock_quantity > 0
        GROUP BY p.id, p.name, c.name, b.name
        ORDER BY MAX(ci.created_at) DESC
        LIMIT 5
    """), {"uid": user_id}).mappings().all()

    rows = conn.execute(text("""
        WITH brand_catalog AS (
            SELECT COUNT(DISTINCT brand_id) FILTER (WHERE brand_id IS NOT NULL) > 1 AS brand_reliable
            FROM products
            WHERE is_active = TRUE
        ),
        signals AS (
            SELECT
                   p.category_id,
                   CASE WHEN bc.brand_reliable THEN p.brand_id ELSE NULL END AS brand_id,
                   c.name AS category_name,
                   CASE WHEN bc.brand_reliable THEN b.name ELSE NULL END AS brand_name,
                   SUM(CASE ui.interaction_type
                       WHEN 'PURCHASE' THEN 5.0
                       WHEN 'ADD_TO_CART' THEN 3.0
                       WHEN 'VIEW' THEN 1.0
                       WHEN 'SEARCH' THEN 0.5
                       ELSE COALESCE(ui.interaction_weight, 1.0)
                   END * EXP(-EXTRACT(EPOCH FROM (NOW() - ui.created_at)) / (86400.0 * 45.0))) AS score
            FROM user_interactions ui
            JOIN products p ON p.id = ui.product_id
            CROSS JOIN brand_catalog bc
            LEFT JOIN categories c ON c.id = p.category_id
            LEFT JOIN brands b ON b.id = p.brand_id
            WHERE ui.user_id = :uid AND ui.product_id IS NOT NULL
            GROUP BY p.category_id, CASE WHEN bc.brand_reliable THEN p.brand_id ELSE NULL END, c.name, CASE WHEN bc.brand_reliable THEN b.name ELSE NULL END

            UNION ALL

            SELECT
                   p.category_id,
                   CASE WHEN bc.brand_reliable THEN p.brand_id ELSE NULL END,
                   c.name,
                   CASE WHEN bc.brand_reliable THEN b.name ELSE NULL END,
                   COUNT(*) * 2.5 AS score
            FROM wishlists w
            JOIN products p ON p.id = w.product_id
            CROSS JOIN brand_catalog bc
            LEFT JOIN categories c ON c.id = p.category_id
            LEFT JOIN brands b ON b.id = p.brand_id
            WHERE w.user_id = :uid
            GROUP BY p.category_id, CASE WHEN bc.brand_reliable THEN p.brand_id ELSE NULL END, c.name, CASE WHEN bc.brand_reliable THEN b.name ELSE NULL END

            UNION ALL

            SELECT
                   p.category_id,
                   CASE WHEN bc.brand_reliable THEN p.brand_id ELSE NULL END,
                   c.name,
                   CASE WHEN bc.brand_reliable THEN b.name ELSE NULL END,
                   SUM(CASE WHEN r.rating >= 4 THEN 2.0 WHEN r.rating <= 2 THEN -2.0 ELSE 0.2 END) AS score
            FROM reviews r
            JOIN products p ON p.id = r.product_id
            CROSS JOIN brand_catalog bc
            LEFT JOIN categories c ON c.id = p.category_id
            LEFT JOIN brands b ON b.id = p.brand_id
            WHERE r.user_id = :uid
            GROUP BY p.category_id, CASE WHEN bc.brand_reliable THEN p.brand_id ELSE NULL END, c.name, CASE WHEN bc.brand_reliable THEN b.name ELSE NULL END
        )
        SELECT category_name, brand_name, SUM(score) AS score
        FROM signals
        GROUP BY category_name, brand_name
        HAVING SUM(score) > 0
        ORDER BY score DESC
        LIMIT 5
    """), {"uid": user_id}).mappings().all()

    bad_rows = conn.execute(text("""
        SELECT DISTINCT p.name
        FROM order_status_history osh
        JOIN orders o ON o.id = osh.order_id
        JOIN order_items oi ON oi.order_id = o.id
        JOIN product_skus ps ON ps.id = oi.product_sku_id
        JOIN products p ON p.id = ps.product_id
        WHERE o.user_id = :uid
          AND osh.status IN ('CANCELLED', 'REFUNDED', 'RETURNED')
        ORDER BY p.name
        LIMIT 5
    """), {"uid": user_id}).scalars().all()

    return_rows = conn.execute(text("""
        SELECT DISTINCT p.name
        FROM order_returns ors
        JOIN orders o ON o.id = ors.order_id
        JOIN order_items oi ON oi.order_id = o.id
        JOIN product_skus ps ON ps.id = oi.product_sku_id
        JOIN products p ON p.id = ps.product_id
        WHERE ors.user_id = :uid
          AND ors.status IN ('PENDING', 'APPROVED')
        ORDER BY p.name
        LIMIT 5
    """), {"uid": user_id}).scalars().all()

    positive_review_rows = conn.execute(text("""
        SELECT p.name, r.rating, c.name AS category_name
        FROM reviews r
        JOIN products p ON p.id = r.product_id
        LEFT JOIN categories c ON c.id = p.category_id
        WHERE r.user_id = :uid AND r.rating >= 4
        ORDER BY r.created_at DESC
        LIMIT 5
    """), {"uid": user_id}).mappings().all()

    chat_rows = conn.execute(text("""
        SELECT cm.message_text
        FROM chatbot_sessions cs
        JOIN chatbot_messages cm ON cm.session_id = cs.id
        WHERE cs.user_id = :uid
          AND cm.sender_type = 'USER'
        ORDER BY cm.created_at DESC
        LIMIT 5
    """), {"uid": user_id}).scalars().all()

    chat_product_rows = conn.execute(text("""
        WITH chat_product_ids AS (
            SELECT CAST(TRIM(pid) AS INTEGER) AS product_id, cm.created_at
            FROM chatbot_sessions cs
            JOIN chatbot_messages cm ON cm.session_id = cs.id
            CROSS JOIN LATERAL regexp_split_to_table(COALESCE(cm.retrieved_product_ids, ''), ',') AS pid
            WHERE cs.user_id = :uid
              AND TRIM(pid) ~ '^[0-9]+$'
        )
        SELECT p.name
        FROM chat_product_ids cpi
        JOIN products p ON p.id = cpi.product_id
        GROUP BY p.id, p.name
        ORDER BY MAX(cpi.created_at) DESC
        LIMIT 5
    """), {"uid": user_id}).scalars().all()

    parts = []
    if profile:
        height = profile["height"]
        weight = profile["weight"]
        if height or weight:
            bmi = _compute_bmi(height, weight)
            bmi_label = _bmi_label(bmi)
            likely_sizes = _estimate_clothing_sizes(height, weight)
            bmi_part = f", BMI khoảng {bmi:.1f} ({bmi_label})" if bmi else ""
            size_part = ", size quần áo ước lượng: " + ", ".join(likely_sizes) if likely_sizes else ""
            parts.append(
                "Hồ sơ cơ thể lấy từ DB theo user_id: "
                f"cao {height or 'chưa có'} cm, nặng {weight or 'chưa có'} kg"
                f"{bmi_part}{size_part}. "
                "Khi tư vấn quần áo/giày, dùng dữ liệu này để gợi ý fit/size một cách thận trọng."
            )
    if rows:
        favs = [
            f"{r['category_name'] or 'danh mục khác'} / {r['brand_name']}"
            if r["brand_name"] else (r["category_name"] or "danh mục khác")
            for r in rows
        ]
        parts.append("Sở thích suy ra từ hành vi gần đây: " + "; ".join(favs))
    if cart_rows:
        cart_text = [
            f"{r['name']} x{int(r['quantity'] or 1)}"
            + (f" ({r['category_name']})" if r["category_name"] else "")
            for r in cart_rows
        ]
        parts.append("Sản phẩm đang trong giỏ hàng, ưu tiên rất cao nếu phù hợp câu hỏi: " + "; ".join(cart_text))
    if positive_review_rows:
        review_text = [
            f"{r['name']} {int(r['rating'])}/5"
            + (f" ({r['category_name']})" if r["category_name"] else "")
            for r in positive_review_rows
        ]
        parts.append("Sản phẩm/danh mục user từng đánh giá tốt: " + "; ".join(review_text))
    if chat_rows:
        cleaned = [str(msg).strip().replace("\n", " ")[:120] for msg in chat_rows if str(msg).strip()]
        if cleaned:
            parts.append("Ý định/sở thích từ lịch sử chat gần đây của chính user này: " + " | ".join(cleaned))
    if chat_product_rows:
        parts.append("Sản phẩm từng được hệ thống gợi ý trong lịch sử chat của user này: " + ", ".join(chat_product_rows))
    if bad_rows:
        parts.append("Nên tránh hoặc giảm ưu tiên sản phẩm từng hủy/hoàn/trả: " + ", ".join(bad_rows))
    if return_rows:
        parts.append("Nên giảm mạnh ưu tiên sản phẩm có yêu cầu đổi/trả: " + ", ".join(return_rows))
    return "\n".join(parts) if parts else "Chưa có đủ lịch sử cá nhân hóa; ưu tiên ý định trong câu hỏi và sản phẩm phổ biến phù hợp."


def _is_body_or_preference_question(message: str) -> bool:
    msg = (message or "").lower()
    tokens = [
        "chiều cao", "cân nặng", "bmi", "body", "dáng người", "vóc dáng",
        "size", "fit", "vừa", "hợp với tôi", "sở thích", "thường thích",
        "dựa trên tôi", "cá nhân hóa", "gợi ý cho tôi",
    ]
    return any(token in msg for token in tokens)


def _merge_ordered_ids(primary: list[int], secondary: list[int], limit: int) -> list[int]:
    seen = set()
    merged = []
    for pid in [*(primary or []), *(secondary or [])]:
        if pid in seen:
            continue
        seen.add(pid)
        merged.append(pid)
        if len(merged) >= limit:
            break
    return merged


def _compute_bmi(height_cm, weight_kg) -> float | None:
    if not height_cm or not weight_kg:
        return None
    height = float(height_cm)
    weight = float(weight_kg)
    if height <= 0 or weight <= 0:
        return None
    height_m = height / 100.0
    return weight / (height_m * height_m)


def _bmi_label(bmi: float | None) -> str:
    if bmi is None:
        return "chưa đủ dữ liệu"
    if bmi < 18.5:
        return "thon/gầy"
    if bmi < 23:
        return "cân đối"
    if bmi < 25:
        return "hơi đầy đặn"
    return "đầy đặn"


def _estimate_clothing_sizes(height_cm, weight_kg) -> list[str]:
    if not height_cm or not weight_kg:
        return []
    height = float(height_cm)
    weight = float(weight_kg)
    bmi = _compute_bmi(height, weight) or 0
    if height < 160 and weight < 55:
        return ["S", "M"]
    if height < 170 and weight < 68:
        return ["M", "S", "L"]
    if height < 180 and weight < 82:
        return ["L", "M", "XL"]
    if bmi >= 27 or weight >= 82:
        return ["XL", "XXL", "L"]
    return ["M", "L"]


def _load_history(conn, session_id: str, exclude_last: bool = True) -> list:
    """
    Kéo lịch sử và chuyển sang định dạng LangChain messages.
    exclude_last=True: bỏ tin nhắn user vừa lưu (sẽ truyền riêng vào HumanMessage).
    """
    rows = conn.execute(text("""
        SELECT sender_type, message_text
        FROM chatbot_messages
        WHERE session_id = :sid
        ORDER BY created_at DESC
        LIMIT :limit
    """), {"sid": session_id, "limit": _HISTORY_LIMIT + 1}).mappings().all()

    # Đảo lại để thứ tự thời gian tăng dần
    rows = list(reversed(rows))

    # Bỏ tin nhắn cuối (là tin nhắn user vừa gửi)
    if exclude_last and rows:
        rows = rows[:-1]

    messages = []
    for r in rows:
        if r["sender_type"] == "USER":
            messages.append(HumanMessage(content=r["message_text"]))
        else:
            messages.append(AIMessage(content=r["message_text"]))
    return messages


def _generate_reply(message: str, context_text: str, personalization_text: str, history: list) -> str:
    """
    Ghép prompt và gọi LLM.
    Dùng LangChain message format để multi-turn hoạt động đúng.
    """
    system_with_context = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"[THÔNG TIN CÁ NHÂN HÓA]\n{personalization_text}\n\n"
        f"[SẢN PHẨM ĐANG CÓ TRONG CỬA HÀNG]\n{context_text}"
    )

    messages = [
        SystemMessage(content=system_with_context),
        *history,
        HumanMessage(content=message),
    ]

    response = _llm.invoke(messages)
    return response.content
