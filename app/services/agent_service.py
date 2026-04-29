import json
import logging
from together import Together
from app.utils.config import Config
from app.services.rag_service import search_similar_products
from app.services.chatbot_service import chat_with_bot
from app.services.recommendation_service import get_recommendations_for_user

logger = logging.getLogger(__name__)

_client = Together(api_key=Config.TOGETHER_API_KEY)
_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# =====================================================================
# TOOL DEFINITIONS — Function Calling schema cho Together AI
# =====================================================================
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Tìm kiếm sản phẩm theo mô tả tự nhiên. "
                "Dùng khi người dùng muốn tìm sản phẩm cụ thể theo tên, loại, thương hiệu, giá, mục đích sử dụng."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Câu mô tả sản phẩm cần tìm. VD: 'giày chạy bộ Nike nam'"},
                    "category_id": {"type": "integer", "description": "Lọc theo danh mục: 1=Giày/Dép, 2=Quần Áo, 3=Phụ Kiện. Bỏ qua nếu không rõ."},
                    "top_k": {"type": "integer", "description": "Số kết quả trả về, mặc định 5, tối đa 20."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_with_bot",
            "description": (
                "Tư vấn sản phẩm dạng hội thoại đa lượt. "
                "Dùng khi người dùng hỏi tư vấn, so sánh, hỏi tiếp nối cuộc trò chuyện trước, "
                "hoặc muốn được tư vấn như nhân viên bán hàng."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Tin nhắn của người dùng."},
                    "session_id": {"type": "string", "description": "ID phiên chat hiện tại (nếu đang trong cuộc trò chuyện)."},
                    "user_id": {"type": "integer", "description": "ID người dùng (nếu đã đăng nhập)."},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": (
                "Gợi ý sản phẩm cá nhân hóa dựa trên lịch sử mua hàng và wishlist của người dùng. "
                "Dùng khi người dùng yêu cầu 'gợi ý cho tôi', 'đề xuất sản phẩm', 'có gì hay cho tôi'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "ID người dùng cần gợi ý."},
                    "limit": {"type": "integer", "description": "Số sản phẩm gợi ý, mặc định 6, tối đa 20."},
                },
                "required": ["user_id"],
            },
        },
    },
]

# Prompt hướng dẫn LLM chọn tool đúng
_SYSTEM_PROMPT = """
Bạn là AI điều phối (orchestrator) cho cửa hàng đồ thể thao.
Dựa vào câu hỏi của người dùng, hãy chọn TOOL phù hợp nhất và trích xuất arguments.

Quy tắc:
- Người dùng muốn tìm sản phẩm cụ thể → search_products
- Người dùng muốn tư vấn, hỏi tiếp, so sánh, hỏi giá → chat_with_bot
- Người dùng muốn gợi ý cá nhân hóa, "có gì hay cho tôi" → get_recommendations
- Chào hỏi đơn thuần ("hello", "xin chào") → chat_with_bot

Trả về JSON với format: {"tool": "tool_name", "arguments": {...}}
Nếu không rõ, mặc định dùng chat_with_bot.
""".strip()


def _call_llm_router(user_message: str, user_id: int | None, session_id: str | None) -> dict:
    """Gọi LLM để chọn tool + arguments."""
    context_info = ""
    if user_id:
        context_info += f"\nuser_id: {user_id}"
    if session_id:
        context_info += f"\nsession_id: {session_id}"

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:{context_info or ' (không có)'}\n"
        f"Câu hỏi người dùng: \"{user_message}\"\n"
        f"Trả về JSON:"
    )

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": "Always return valid JSON only. No explanations, no markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def _execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute tool và trả về kết quả."""
    if tool_name == "search_products":
        query = arguments.get("query", "")
        top_k = min(int(arguments.get("top_k", 5)), 20)
        category_id = arguments.get("category_id")
        product_ids = search_similar_products(
            query_text=query,
            target_category_id=category_id,
            top_k=top_k,
        )
        return {"tool": "search_products", "product_ids": product_ids, "query": query}

    elif tool_name == "chat_with_bot":
        message = arguments.get("message", "")
        sid = arguments.get("session_id")
        uid = arguments.get("user_id")
        result = chat_with_bot(session_id=sid, user_id=uid, message=message)
        return {
            "tool": "chat_with_bot",
            "session_id": result["session_id"],
            "reply": result["reply"],
            "suggested_product_ids": result["suggested_product_ids"],
        }

    elif tool_name == "get_recommendations":
        uid = arguments.get("user_id")
        limit = min(int(arguments.get("limit", 6)), 20)
        product_ids = get_recommendations_for_user(user_id=uid, limit=limit)
        return {"tool": "get_recommendations", "user_id": uid, "product_ids": product_ids}

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def handle_user_request(
    message: str,
    user_id: int | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Unified AI endpoint với Function Calling.

    Flow:
      1. LLM phân tích intent → chọn TOOL + arguments
      2. Backend execute tool tương ứng
      3. Trả về kết quả
    """
    try:
        # ── Bước 1: LLM chọn tool ──────────────────────────────────────
        route = _call_llm_router(message, user_id, session_id)
        tool_name = route.get("tool", "chat_with_bot")
        arguments = route.get("arguments", {})

        # Bổ sung context nếu LLM không trích xuất đủ
        if user_id and "user_id" not in arguments:
            arguments["user_id"] = user_id
        if session_id and "session_id" not in arguments:
            arguments["session_id"] = session_id

        logger.info(f"🤖 LLM chọn tool='{tool_name}' args={arguments}")

        # ── Bước 2: Execute tool ───────────────────────────────────────
        result = _execute_tool(tool_name, arguments)
        logger.info(f"✅ Tool '{tool_name}' thực thi thành công")
        return result

    except Exception as e:
        logger.error(f"❌ Agent lỗi: {e}", exc_info=True)
        # Fallback về chat nếu LLM routing lỗi
        return chat_with_bot(session_id=session_id, user_id=user_id, message=message)
