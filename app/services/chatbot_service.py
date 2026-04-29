import logging
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import text
from app.services.rag_service import search_similar_products
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
Chỉ tư vấn sản phẩm dựa trên thông tin được cung cấp trong phần [SẢN PHẨM].
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
            if not session_id:
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
            context_text, retrieved_ids = _retrieve_context(conn, message)

            # ── 4. Kéo lịch sử chat ───────────────────────────────────────
            history_messages = _load_history(conn, session_id, exclude_last=True)

        # ── 5. Gọi LLM ───────────────────────────────────────────────────
        bot_reply = _generate_reply(
            message=message,
            context_text=context_text,
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
        }

    except Exception as e:
        logger.error(f"❌ Chatbot lỗi session={session_id}: {e}", exc_info=True)
        raise


# ── PRIVATE HELPERS ──────────────────────────────────────────────────────────

def _retrieve_context(conn, message: str) -> tuple[str, list[int]]:
    """
    Sử dụng sức mạnh của Hybrid RAG thay vì Vector thuần.
    """
    # Gọi trực tiếp hàm RAG đã được tối ưu siêu việt của bạn
    # Hàm này tự động gọi Intent Service, lọc SQL giá, màu sắc và Rerank
    retrieved_ids = search_similar_products(query_text=message, top_k=_RAG_TOP_K)
    
    if not retrieved_ids:
        return "Không có sản phẩm phù hợp.", []

    # Lấy thông tin chi tiết của các ID đã lọc để nạp vào Prompt
    rows = conn.execute(text("""
        SELECT id, name, base_price, description 
        FROM products 
        WHERE id = ANY(:pids)
    """), {"pids": retrieved_ids}).mappings().all()

    context_parts = []
    for r in rows:
        context_parts.append(f"- ID {r['id']}: {r['name']} | Giá: {r['base_price']} | Mô tả: {r['description']}")
        
    context_text = "\n".join(context_parts)
    return context_text, retrieved_ids


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


def _generate_reply(message: str, context_text: str, history: list) -> str:
    """
    Ghép prompt và gọi LLM.
    Dùng LangChain message format để multi-turn hoạt động đúng.
    """
    system_with_context = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"[SẢN PHẨM ĐANG CÓ TRONG CỬA HÀNG]\n{context_text}"
    )

    messages = [
        SystemMessage(content=system_with_context),
        *history,
        HumanMessage(content=message),
    ]

    response = _llm.invoke(messages)
    return response.content