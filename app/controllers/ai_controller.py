import logging
from flask import Blueprint, jsonify, request
from app.services.rag_service import search_similar_products
from app.services.chatbot_service import chat_with_bot
from app.services.recommendation_service import get_recommendations_for_user

logger = logging.getLogger(__name__)

ai_bp = Blueprint("ai_controller", __name__)


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _ok(data: dict, status: int = 200):
    return jsonify({"status": "success", **data}), status


def _err(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


def _get_json():
    """Trả về (data, error_response). Nếu body không phải JSON → error_response khác None."""
    data = request.get_json(silent=True)
    if not data:
        return None, _err("Body phải là JSON hợp lệ.")
    return data, None


# ── HEALTH ───────────────────────────────────────────────────────────────────

@ai_bp.route("/health", methods=["GET"])
def health_check():
    return _ok({"message": "AI Service is running!"})


# ── SEARCH ───────────────────────────────────────────────────────────────────

@ai_bp.route("/search", methods=["POST"])
def smart_search():
    """
    Tìm kiếm sản phẩm bằng AI (RAG + intent analysis).

    Body:
        query       (str, bắt buộc)  — câu tìm kiếm tự nhiên
        category_id (int, tuỳ chọn) — lọc cứng theo danh mục từ UI
        top_k       (int, tuỳ chọn) — số kết quả trả về, mặc định 5, tối đa 20

    Response:
        product_ids (list[int])
        query       (str)
    """
    data, err = _get_json()
    if err:
        return err

    query = (data.get("query") or "").strip()
    if not query:
        return _err("Trường 'query' không được để trống.")

    # Validate top_k: phải là số nguyên dương, tối đa 20
    try:
        top_k = min(int(data.get("top_k", 5)), 20)
        if top_k < 1:
            raise ValueError
    except (ValueError, TypeError):
        return _err("Trường 'top_k' phải là số nguyên dương.")

    # Validate category_id nếu có
    category_id = data.get("category_id")
    if category_id is not None:
        try:
            category_id = int(category_id)
        except (ValueError, TypeError):
            return _err("Trường 'category_id' phải là số nguyên.")

    try:
        product_ids = search_similar_products(
            query_text=query,
            target_category_id=category_id,
            top_k=top_k,
        )
        logger.info(f"[SEARCH] query='{query}' cat={category_id} top_k={top_k} → {len(product_ids)} kết quả")
        return _ok({"query": query, "product_ids": product_ids})

    except Exception as e:
        logger.error(f"[SEARCH] Lỗi nội bộ: {e}", exc_info=True)
        return _err("Đã xảy ra lỗi khi tìm kiếm. Vui lòng thử lại.", 500)


# ── CHAT ─────────────────────────────────────────────────────────────────────

@ai_bp.route("/chat", methods=["POST"])
def smart_chat():
    """
    Chatbot tư vấn sản phẩm (RAG + multi-turn memory).

    Body:
        message     (str, bắt buộc)  — tin nhắn của người dùng
        session_id  (str, tuỳ chọn) — ID phiên chat hiện tại;
                                       để trống khi bắt đầu cuộc trò chuyện mới
        user_id     (int, tuỳ chọn) — ID người dùng (nếu đã đăng nhập)

    Response:
        session_id          (str)
        reply               (str)
        suggested_product_ids (list[int])
    """
    data, err = _get_json()
    if err:
        return err

    message = (data.get("message") or "").strip()
    if not message:
        return _err("Trường 'message' không được để trống.")

    session_id = data.get("session_id") or None  # None → service tự tạo mới

    user_id = data.get("user_id")
    if user_id is not None:
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return _err("Trường 'user_id' phải là số nguyên.")

    try:
        result = chat_with_bot(
            session_id=session_id,
            user_id=user_id,
            message=message,
        )
        logger.info(f"[CHAT] session={result['session_id']} user={user_id} → reply={result['reply'][:60]}...")
        return _ok({
            "session_id":            result["session_id"],
            "reply":                 result["reply"],
            "suggested_product_ids": result["suggested_product_ids"],
        })

    except Exception as e:
        logger.error(f"[CHAT] Lỗi nội bộ session={session_id}: {e}", exc_info=True)
        return _err("Đã xảy ra lỗi khi xử lý chat. Vui lòng thử lại.", 500)


# ── RECOMMEND ────────────────────────────────────────────────────────────────

@ai_bp.route("/recommend", methods=["POST"])
def recommend():
    """
    Gợi ý sản phẩm cá nhân hóa dựa trên lịch sử mua và wishlist.

    Body:
        user_id (int, bắt buộc) — ID người dùng cần gợi ý
        limit   (int, tuỳ chọn) — số sản phẩm gợi ý, mặc định 6, tối đa 20

    Response:
        product_ids (list[int])
        user_id     (int)

    Lưu ý: Endpoint này sử dụng recommendation_service (seed từ orders + wishlist).
    KHÔNG dùng search_similar_products vì service đó dành cho tìm kiếm theo keyword.
    """
    data, err = _get_json()
    if err:
        return err

    # user_id bắt buộc — không có user thì không thể cá nhân hóa
    raw_user_id = data.get("user_id")
    if raw_user_id is None:
        return _err("Trường 'user_id' là bắt buộc cho tính năng gợi ý.")
    try:
        user_id = int(raw_user_id)
    except (ValueError, TypeError):
        return _err("Trường 'user_id' phải là số nguyên.")

    try:
        limit = min(int(data.get("limit", 6)), 20)
        if limit < 1:
            raise ValueError
    except (ValueError, TypeError):
        return _err("Trường 'limit' phải là số nguyên dương.")

    try:
        product_ids = get_recommendations_for_user(user_id=user_id, limit=limit)
        logger.info(f"[RECOMMEND] user={user_id} limit={limit} → {len(product_ids)} gợi ý")
        return _ok({"user_id": user_id, "product_ids": product_ids})

    except Exception as e:
        logger.error(f"[RECOMMEND] Lỗi nội bộ user={user_id}: {e}", exc_info=True)
        return _err("Đã xảy ra lỗi khi tạo gợi ý. Vui lòng thử lại.", 500)