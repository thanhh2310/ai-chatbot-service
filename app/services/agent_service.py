import json
import logging
import uuid
import re
from typing import Generator
from together import Together
from sqlalchemy import text
from app.utils.config import Config
from app.services.rag_service import search_similar_products
from app.services.recommendation_service import get_recommendations_for_user
from app.services.db_service import db

engine = db.get_engine()
logger = logging.getLogger(__name__)

_client = Together(api_key=Config.TOGETHER_API_KEY)
_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# =====================================================================
# TOOL DEFINITIONS
# =====================================================================
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Tìm kiếm sản phẩm theo mô tả tự nhiên (tên, loại, thương hiệu, giá, mục đích).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Mô tả sản phẩm cần tìm"},
                    "category_id": {"type": "integer", "description": "Lọc danh mục: 1=Giày/Dép, 2=Quần Áo, 3=Phụ Kiện"},
                    "top_k": {"type": "integer", "description": "Số kết quả, mặc định 5, tối đa 20"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_with_bot",
            "description": "Tư vấn sản phẩm dạng hội thoại đa lượt. Dùng khi hỏi tư vấn, so sánh, hỏi tiếp, chào hỏi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Tin nhắn người dùng"},
                    "session_id": {"type": "string", "description": "ID phiên chat hiện tại"},
                    "user_id": {"type": "integer", "description": "ID người dùng"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": "Gợi ý sản phẩm cá nhân hóa dựa trên lịch sử mua hàng và wishlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "ID người dùng cần gợi ý"},
                    "limit": {"type": "integer", "description": "Số sản phẩm gợi ý, mặc định 6, tối đa 20"},
                },
                "required": ["user_id"],
            },
        },
    },
]

_ROUTER_SYSTEM = """
Bạn là AI điều phối (orchestrator) cho cửa hàng đồ thể thao.
Dựa vào câu hỏi, chọn TOOL phù hợp nhất và trích xuất arguments.

Quy tắc chọn tool:
- Tìm sản phẩm cụ thể, hỏi tên sản phẩm → search_products
- Tư vấn, hỏi tiếp, so sánh, chào hỏi, câu hỏi thường ngày → chat_with_bot
- Câu hỏi cá nhân hóa như "hợp với tôi không", "body type", "size nào", "tôi thường thích gì" → chat_with_bot và luôn truyền user_id nếu có
- Câu hỏi dựa trên chiều cao/cân nặng/BMI, lịch sử chat, giỏ hàng, review, wishlist, lịch sử mua/đổi trả → chat_with_bot và luôn truyền user_id nếu có
- "gợi ý cho tôi", "đề xuất sản phẩm", "có gì hay cho tôi" khi không có ràng buộc cụ thể → get_recommendations

Quy tắc bắt buộc về arguments cho search_products:
- LUÔN dùng key "query" (KHÔNG được dùng product_name, keyword, name, search_query hay bất kỳ tên nào khác).
- Ví dụ đúng: {"tool": "search_products", "arguments": {"query": "Giày Nike chạy bộ"}}

Trả về JSON: {"tool": "tool_name", "arguments": {...}}
""".strip()


def _call_llm_router(user_message: str, user_id: int | None, session_id: str | None) -> dict:
    context_info = ""
    if user_id: context_info += f"\nuser_id: {user_id}"
    if session_id: context_info += f"\nsession_id: {session_id}"

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": "Always return valid JSON only. No explanations, no markdown."},
            {"role": "user", "content": f"{_ROUTER_SYSTEM}\n\nContext:{context_info or ' (không có)'}\nCâu hỏi: \"{user_message}\"\nTrả về JSON:"},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def _fetch_product_details(product_ids: list[int]) -> list[dict]:
    if not product_ids:
        return []
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.id, p.name, p.base_price, p.description, p.slug,
                   b.name AS brand_name, c.name AS category_name,
                   pi.image_url AS thumbnail_url,
                   COALESCE(attrs.in_stock_attributes, '') AS in_stock_attributes
            FROM products p
            LEFT JOIN brands b ON p.brand_id = b.id
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN LATERAL (
                SELECT image_url FROM product_images
                WHERE product_id = p.id AND is_thumbnail = TRUE
                ORDER BY display_order ASC
                LIMIT 1
            ) pi ON TRUE
            LEFT JOIN LATERAL (
                SELECT string_agg(DISTINCT a.name || ': ' || av.value, ', ' ORDER BY a.name || ': ' || av.value) AS in_stock_attributes
                FROM product_skus sk
                JOIN sku_values sv ON sv.product_sku_id = sk.id
                JOIN attribute_values av ON av.id = sv.attribute_value_id
                JOIN attributes a ON a.id = av.attribute_id
                WHERE sk.product_id = p.id
                  AND sk.is_active = TRUE
                  AND sk.stock_quantity > 0
            ) attrs ON TRUE
            WHERE p.id = ANY(:pids) AND p.is_active = TRUE
              AND EXISTS (
                  SELECT 1
                  FROM product_skus sk
                  WHERE sk.product_id = p.id
                    AND sk.is_active = TRUE
                    AND sk.stock_quantity > 0
              )
        """), {"pids": product_ids}).mappings().all()
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id[pid] for pid in product_ids if pid in by_id]


def _format_products(products: list[dict]) -> str:
    if not products:
        return "Không có sản phẩm phù hợp."
    parts = []
    for p in products:
        link = f"[{p['name']}](http://localhost:8080/api/products/{p['id']})"
        img = f"![{p['name']}]({p['thumbnail_url']})" if p.get('thumbnail_url') else ""
        parts.append(
            f"{img}\n{link} | Giá: {p['base_price']:,.0f} VNĐ"
            f"{' | ' + p['brand_name'] if p['brand_name'] else ''}"
            f"{' | ' + p['category_name'] if p['category_name'] else ''}"
            f"{' | Thuộc tính còn hàng: ' + p['in_stock_attributes'] if p.get('in_stock_attributes') else ''}"
            f"{' | ' + p['description'][:120] if p['description'] else ''}"
        )
    return "\n".join(parts)


# =====================================================================
# TOOL EXECUTORS — trả về product context để LLM sinh reply
# =====================================================================

def _exec_search(arguments: dict) -> tuple[str, list[dict]]:
    # Robust: thử nhiều key alias mà LLM có thể sinh ra thay vì 'query'
    _QUERY_ALIASES = ["query", "product_name", "keyword", "name", "search_query", "product"]
    query = ""
    for alias in _QUERY_ALIASES:
        val = arguments.get(alias, "")
        if val:
            query = str(val)
            if alias != "query":
                logger.warning(f"⚠️ LLM dùng key alias '{alias}' thay vì 'query'. Đã tự động fix.")
            break

    top_k = min(int(arguments.get("top_k", 5)), 20)
    category_id = arguments.get("category_id")
    product_ids = search_similar_products(
        query_text=query,
        target_category_id=category_id,
        top_k=top_k,
        user_id=arguments.get("user_id"),
    )
    products = _fetch_product_details(product_ids)
    return _format_products(products), products


def _exec_recommend(arguments: dict) -> tuple[str, list[dict]]:
    uid = arguments.get("user_id")
    limit = min(int(arguments.get("limit", 6)), 20)
    product_ids = get_recommendations_for_user(user_id=uid, limit=limit)
    products = _fetch_product_details(product_ids)
    product_text = _format_products(products)
    if uid:
        from app.services.chatbot_service import _build_personalization_context
        with engine.connect() as conn:
            personalization_text = _build_personalization_context(conn, uid)
        product_text = f"[THÔNG TIN CÁ NHÂN HÓA]\n{personalization_text}\n\n[SẢN PHẨM GỢI Ý]\n{product_text}"
    return product_text, products


def _exec_chat(arguments: dict, user_message: str) -> tuple[str, list[dict], str, list]:
    """Trả về: (context_text, products, session_id, history_messages)"""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from app.services.chatbot_service import (
        _build_personalization_context,
        _ensure_session_access,
        _is_body_or_preference_question,
        _merge_ordered_ids,
    )

    sid = arguments.get("session_id")
    uid = arguments.get("user_id")

    with engine.connect() as conn:
        if sid:
            _ensure_session_access(conn, sid, uid)
        else:
            sid = str(uuid.uuid4())
            conn.execute(text("INSERT INTO chatbot_sessions (id, user_id) VALUES (:id, :uid)"), {"id": sid, "uid": uid})
            conn.commit()

        conn.execute(text("INSERT INTO chatbot_messages (session_id, sender_type, message_text) VALUES (:sid, 'USER', :msg)"),
                     {"sid": sid, "msg": user_message})
        conn.commit()

        # RAG context
        retrieved_ids = search_similar_products(query_text=user_message, top_k=4, user_id=uid)
        if uid and _is_body_or_preference_question(user_message):
            recommendation_ids = get_recommendations_for_user(user_id=uid, limit=4)
            retrieved_ids = _merge_ordered_ids(retrieved_ids, recommendation_ids, 4)
        if retrieved_ids:
            rows = conn.execute(text("""
                SELECT p.id, p.name, p.base_price, p.description, p.slug,
                       pi.image_url AS thumbnail_url,
                       COALESCE(attrs.in_stock_attributes, '') AS in_stock_attributes
                FROM products p
                LEFT JOIN LATERAL (
                    SELECT image_url FROM product_images
                    WHERE product_id = p.id AND is_thumbnail = TRUE
                    ORDER BY display_order ASC LIMIT 1
                ) pi ON TRUE
                LEFT JOIN LATERAL (
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
            rows_by_id = {r["id"]: r for r in rows}
            filtered_ids = []
            for pid in retrieved_ids:
                r = rows_by_id.get(pid)
                if not r:
                    continue
                filtered_ids.append(r["id"])
                link = f"[{r['name']}](http://localhost:8080/api/products/{r['id']})"
                img = f"![{r['name']}]({r['thumbnail_url']})" if r.get('thumbnail_url') else ""
                context_parts.append(
                    f"{img}\n{link} | Giá: {r['base_price']} | "
                    f"Thuộc tính còn hàng: {r['in_stock_attributes'] or 'chưa có'} | "
                    f"Mô tả: {r['description']}"
                )
            retrieved_ids = filtered_ids
            context_text = "\n".join(context_parts)
        else:
            context_text = "Không có sản phẩm phù hợp."

        personalization_text = _build_personalization_context(conn, uid)
        context_text = f"[THÔNG TIN CÁ NHÂN HÓA]\n{personalization_text}\n\n[SẢN PHẨM]\n{context_text}"

        # History
        rows = conn.execute(text("""
            SELECT sender_type, message_text FROM chatbot_messages
            WHERE session_id = :sid ORDER BY created_at DESC LIMIT 7
        """), {"sid": sid}).mappings().all()
        rows = list(reversed(rows))
        if rows: rows = rows[:-1]
        history = []
        for r in rows:
            history.append(HumanMessage(content=r["message_text"]) if r["sender_type"] == "USER" else AIMessage(content=r["message_text"]))

    products = _fetch_product_details(retrieved_ids)
    return context_text, products, sid, history


# =====================================================================
# STREAMING GENERATOR
# =====================================================================

_CHAT_SYSTEM_PROMPT = """
Bạn là nhân viên tư vấn bán hàng thể thao chuyên nghiệp, thân thiện.
Xưng "em", gọi khách là "anh/chị".
Chỉ tư vấn dựa trên thông tin trong phần [SẢN PHẨM] và dữ liệu cá nhân hóa đã được hệ thống truy xuất.
Khi khách hỏi về size, màu hoặc thuộc tính sản phẩm, chỉ trả lời theo "Thuộc tính còn hàng" trong [SẢN PHẨM].
Với câu hỏi về sở thích, body type, size, sản phẩm thường thích, hãy giải thích theo lịch sử hành vi/profile nếu có.
Không khẳng định chắc chắn size chỉ từ chiều cao/cân nặng; hãy nói đó là ước lượng và nhắc khách kiểm tra bảng size nếu cần.
Nếu không có sản phẩm phù hợp, nói: "Em chưa tìm được sản phẩm phù hợp trong cửa hàng, anh/chị có thể mô tả rõ hơn không ạ?"
Không bịa thông tin giá, tên sản phẩm, hoặc tính năng ngoài ngữ cảnh.
Trả lời ngắn gọn, tối đa 3-4 câu.

""".strip()

_SEARCH_PROMPT = """Bạn là trợ lý bán hàng thể thao. Dựa vào [SẢN PHẨM TÌM ĐƯỢC], hãy giới thiệu sản phẩm cho khách.
Mô tả ngắn gọn, nêu giá, thương hiệu, đặc điểm nổi bật.
Nếu không có sản phẩm phù hợp, nói: "Em chưa tìm được sản phẩm phù hợp, anh/chị có thể mô tả rõ hơn không ạ?"
Trả lời tự nhiên, xưng "em", gọi "anh/chị". Không dùng JSON.

""".strip()

_RECOMMEND_PROMPT = """Bạn là trợ lý bán hàng thể thao. Dựa vào [SẢN PHẨM GỢI Ý], hãy giới thiệu sản phẩm cho khách.
Giải thích ngắn gọn vì sao gợi ý các sản phẩm này dựa trên dữ liệu cá nhân hóa: chiều cao/cân nặng/BMI nếu có, giỏ hàng, lịch sử mua, wishlist, review, hành vi tương tác, đổi/trả và lịch sử chat của đúng user hiện tại.
Nếu thiếu chiều cao/cân nặng, nói rõ là chưa đủ dữ liệu body metric thay vì tự đoán.
Nêu giá, thương hiệu, đặc điểm nổi bật của từng sản phẩm.
Nếu không có sản phẩm, nói: "Em chưa có sản phẩm nào để gợi ý lúc này ạ."
Trả lời tự nhiên, xưng "em", gọi "anh/chị". Không dùng JSON.

""".strip()


def _stream_reply(messages: list[dict], temperature: float = 0.3) -> Generator[str, None, None]:
    """Stream reply từ Together AI."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def generate_streaming_reply(
    tool_name: str,
    user_message: str,
    product_context: str,
    products: list[dict],
    history: list | None = None,
    session_id: str | None = None,
) -> Generator[str, None, None]:
    """Sinh câu trả lời streaming kèm thông tin sản phẩm."""

    if tool_name == "chat_with_bot":
        from langchain_core.messages import SystemMessage, HumanMessage
        system_with_context = f"{_CHAT_SYSTEM_PROMPT}\n\n[SẢN PHẨM]\n{product_context}"
        messages_dict = [{"role": "system", "content": system_with_context}]
        for msg in history or []:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages_dict.append({"role": role, "content": msg.content})
        messages_dict.append({"role": "user", "content": user_message})
    else:
        if tool_name == "search_products":
            system = _SEARCH_PROMPT
        else:
            system = _RECOMMEND_PROMPT

        messages_dict = [
            {"role": "system", "content": f"{system}\n\n[SẢN PHẨM]\n{product_context}"},
            {"role": "user", "content": user_message},
        ]

    yield from _stream_reply(messages_dict)


def _requires_personalized_chat(message: str) -> bool:
    msg = (message or "").lower()
    tokens = [
        "chiều cao", "cân nặng", "bmi", "body", "dáng người", "vóc dáng",
        "size", "fit", "vừa với tôi", "hợp với tôi", "sở thích", "thường thích",
        "dựa trên tôi", "dựa trên thông tin của tôi", "cá nhân hóa",
        "giỏ hàng", "wishlist", "review", "đánh giá của tôi", "lịch sử chat",
        "lịch sử mua", "đã mua", "đổi trả", "trả hàng",
    ]
    return any(token in msg for token in tokens)


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def handle_user_request_stream(
    message: str,
    user_id: int | None = None,
    session_id: str | None = None,
) -> Generator[dict, None, None]:
    """
    Flow:
      1. LLM routing (non-stream, fast)
      2. Execute tool → get product context
      3. Stream LLM reply với product info
    """
    try:
        # Bước 1: LLM chọn tool
        route = _call_llm_router(message, user_id, session_id)
        tool_name = route.get("tool", "chat_with_bot")
        arguments = route.get("arguments", {})
        if user_id and "user_id" not in arguments: arguments["user_id"] = user_id
        if session_id and "session_id" not in arguments: arguments["session_id"] = session_id
        if user_id and _requires_personalized_chat(message):
            tool_name = "chat_with_bot"
            arguments["message"] = message
            arguments["user_id"] = user_id

        logger.info(f"🤖 LLM chọn tool='{tool_name}' args={arguments}")

        # Bước 2: Execute tool → lấy product context
        if tool_name == "search_products":
            product_context, products = _exec_search(arguments)
            history = None
        elif tool_name == "get_recommendations":
            product_context, products = _exec_recommend(arguments)
            history = None
        else:  # chat_with_bot
            product_context, products, session_id, history = _exec_chat(arguments, message)

        # Bước 3: Stream reply
        reply_chunks = []
        for chunk in generate_streaming_reply(
            tool_name=tool_name,
            user_message=message,
            product_context=product_context,
            products=products,
            history=history,
            session_id=session_id,
        ):
            reply_chunks.append(chunk)
            yield {"type": "chunk", "content": chunk, "tool": tool_name, "session_id": session_id}

        full_reply = "".join(reply_chunks)

        # Lưu reply vào DB nếu là chat
        if tool_name == "chat_with_bot" and session_id:
            with engine.connect() as conn:
                product_ids = [p["id"] for p in products]
                conn.execute(text("""
                    INSERT INTO chatbot_messages (session_id, sender_type, message_text, retrieved_product_ids)
                    VALUES (:sid, 'BOT', :msg, :pids)
                """), {"sid": session_id, "msg": full_reply, "pids": ",".join(map(str, product_ids))})
                conn.commit()

        # Gửi event done
        yield {
            "type": "done",
            "content": full_reply,
            "tool": tool_name,
            "session_id": session_id,
            "products": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "slug": p.get("slug"),
                    "base_price": float(p["base_price"]),
                    "thumbnail_url": p.get("thumbnail_url"),
                    "brand": p.get("brand_name"),
                    "category": p.get("category_name"),
                    "description": p.get("description"),
                }
                for p in products
            ],
        }

    except PermissionError as e:
        logger.warning(f"⚠️ Agent từ chối truy cập session: {e}")
        yield {"type": "error", "content": "Bạn không có quyền truy cập lịch sử chat này."}
    except Exception as e:
        logger.error(f"❌ Agent lỗi: {e}", exc_info=True)
        yield {"type": "error", "content": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại."}
