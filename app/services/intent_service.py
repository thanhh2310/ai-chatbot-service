import json
import logging
import re
from typing import Optional, List
from together import Together
from pydantic import BaseModel, Field
from app.utils.config import Config

logger = logging.getLogger(__name__)

_client = Together(api_key=Config.TOGETHER_API_KEY)
_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

class QueryIntent(BaseModel):
    search_keywords: str = Field(
        description=(
            "Từ khóa cốt lõi TÍCH CỰC, đã loại bỏ thông tin phủ định. "
            "Ưu tiên danh từ và tính từ mô tả sản phẩm. "
            "VD: 'giày chạy bộ Nike nam nhẹ thoáng khí'."
        )
    )
    max_budget: Optional[int] = Field(
        description="Ngân sách tối đa người dùng đề cập (VNĐ). null nếu không đề cập.",
        default=None,
    )
    category_id: Optional[int] = Field(
        description=(
            "ID danh mục: 1 = Giày/Dép (khi thấy 'đôi', 'giày', 'dép', 'sneaker'), "
            "2 = Quần Áo (áo, quần, bộ quần áo, đồ tập), "
            "3 = Phụ Kiện (túi, mũ, vớ, balo). "
            "null nếu không rõ hoặc câu hỏi chung chung."
        ),
        default=None,
    )
    brand_name: Optional[str] = Field(
        description=(
            "Tên hãng chính xác (Nike, Adidas, Puma, Under Armour, Li-Ning, New Balance...). "
            "Lưu ý quy đổi từ lóng: 'das' = Adidas, 'nai' = Nike, 'ua' = Under Armour. "
            "null nếu không đề cập."
        ),
        default=None,
    )
    sport_type: Optional[str] = Field(
        description="Môn thể thao cụ thể: 'chạy bộ', 'gym', 'bóng đá', 'bóng rổ', 'tennis', 'yoga'...",
        default=None,
    )
    gender: Optional[str] = Field(
        description="Giới tính: 'nam', 'nữ', 'unisex'. Mặc định 'unisex' nếu không nhắc tới.",
        default="unisex",
    )
    color_preference: Optional[str] = Field(
        description=(
            "Màu sắc mong muốn nếu có (đen, trắng, đỏ, xanh...). "
            "null nếu không đề cập. KHÔNG điền màu từ excluded_keywords."
        ),
        default=None,
    )
    price_tier: Optional[str] = Field(
        description=(
            "Phân khúc giá: 'binh_dan' (< 500k), 'tam_trung' (500k–1.5tr), "
            "'cao_cap' (> 1.5tr). null nếu không rõ."
        ),
        default=None,
    )
    excluded_keywords: List[str] = Field(
        description=(
            "Các từ khóa khách KHÔNG muốn — màu sắc, chất liệu, đặc tính bị phủ định. "
            "VD: ['đen', 'xám', 'chạy bộ']. Mảng rỗng nếu không có."
        ),
        default=[],
    )

# Prompt cải tiến: thêm few-shot examples để giảm hallucination
_PROMPT_TEMPLATE = """
Bạn là AI trích xuất ý định mua hàng cho website thương mại điện tử. Phân tích câu hỏi và trả về JSON.

Quy tắc quan trọng:
- "đôi" → category_id = 1 (Giày)
- Chỉ điền category_id khi câu hỏi thể hiện rõ danh mục; nếu không chắc, để null để hệ thống tìm theo ngữ nghĩa và thuộc tính.
- "không lấy X", "trừ màu X", "đừng X" → thêm X vào excluded_keywords, KHÔNG vào search_keywords
- Nếu có màu sắc TÍCH CỰC (muốn mua) → điền color_preference
- Ước tính price_tier từ max_budget nếu không nói rõ
- Giữ lại các thuộc tính như size, màu, chất liệu, form dáng, mục đích sử dụng trong search_keywords.
- Nếu người dùng hỏi ngành hàng không thuộc giày/quần áo/phụ kiện, để category_id = null và mô tả đúng nhu cầu trong search_keywords.

Ví dụ:
Input: "giày Nike chạy bộ nam 1 triệu không lấy màu đen"
Output: {{search_keywords: "giày Nike chạy bộ nam", max_budget: 1000000, category_id: 1,
          brand_name: "Nike", sport_type: "chạy bộ", gender: "nam",
          color_preference: null, price_tier: "tam_trung", excluded_keywords: ["đen"]}}

Input: "quần áo gym nam không màu đen hay xám"
Output: {{search_keywords: "quần áo gym nam", max_budget: null, category_id: 2,
          brand_name: null, sport_type: "gym", gender: "nam",
          color_preference: null, price_tier: null, excluded_keywords: ["đen", "xám"]}}

Input: "áo gym nữ màu hồng giá rẻ"
Output: {{search_keywords: "áo gym nữ thoáng khí", max_budget: null, category_id: 2,
          brand_name: null, sport_type: "gym", gender: "nữ",
          color_preference: "hồng", price_tier: "binh_dan", excluded_keywords: []}}

Câu hỏi của khách: "{query}"
"""


def _regex_fallback_extract(query_text: str) -> dict:
    """Xử lý dự phòng bằng Regex khi Gemini API gặp sự cố."""
    q = query_text.lower()
    max_budget = None

    # Bắt giá tiền (Hỗ trợ tiếng lóng "củ", "triệu rưỡi")
    if m := re.search(r'(\d+)\s*(triệu|củ)\s*rưỡi', q):
        max_budget = int(m.group(1)) * 1_000_000 + 500_000
    elif m := re.search(r'(\d+(?:[.,]\d+)?)\s*(triệu|tr|củ)\b', q):
        max_budget = int(float(m.group(1).replace(',', '.')) * 1_000_000)
    elif m := re.search(r'(\d+)\s*k\b', q):
        max_budget = int(m.group(1)) * 1_000

    # Loại bỏ phần phủ định ra khỏi search_keywords và lấy màu excluded
    excluded: list[str] = []
    # Tìm đoạn "không lấy màu X" (dừng lại khi gặp dấu phẩy, chấm, hoặc chấm than)
    neg_match = re.search(r'(không\s+màu|không\s+lấy|không\s+thích|trừ|đừng\s+lấy)\s+(màu\s+)?([^.,;!]+)', q)
    if neg_match:
        phrase = neg_match.group(3)
        for color in ["đen", "trắng", "đỏ", "xanh", "xám", "vàng", "hồng", "tím", "cam", "nâu"]:
            if re.search(rf'(?:^|\s|\W|_){color}(?:$|\s|\W|_)', phrase, re.IGNORECASE | re.UNICODE):
                if color not in excluded:
                    excluded.append(color)
        
        # Bỏ đi đoạn mang từ chối ra khỏi chuỗi gốc để phần phía sau (nếu có) không bị mất
        q = q.replace(neg_match.group(0), " ")

    clean_q = q.strip().replace("đôi", "giày")
    
    brand_mapping = {
        "nike": "nike", "nai": "nike",
        "adidas": "adidas", "das": "adidas",
        "under armour": "under armour", "ua": "under armour",
        "puma": "puma",
        "li-ning": "li-ning", "lining": "li-ning",
        "new balance": "new balance", "nb": "new balance"
    }
    brand = None
    for key, val in brand_mapping.items():
        # Dùng Regex để tránh việc chữ "của" bị nhận nhầm thành hãng "ua"
        if re.search(rf'(?:^|\s|\W|_){key}(?:$|\s|\W|_)', clean_q, re.IGNORECASE):
            brand = val
            break
    
    color_pref = None
    if m := re.search(r'màu\s+(đen|trắng|đỏ|xanh|xám|vàng|hồng|tím|cam|nâu)', clean_q):
        color_pref = m.group(1)

    has_shoes = "giày" in clean_q or "dép" in clean_q
    has_apparel = any(w in clean_q for w in ["áo", "quần", "bộ", "tập gym"])
    cat_id = None
    if has_shoes and not has_apparel:
        cat_id = 1
    elif has_apparel and not has_shoes:
        cat_id = 2

    sport = None
    if any(w in q for w in ["đá bóng", "bóng đá"]):
        sport = "bóng đá"
    elif any(w in q for w in ["chạy bộ", "chạy"]):
        sport = "chạy bộ"
    elif any(w in q for w in ["gym", "thể hình", "tập tạ"]):
        sport = "gym"
    else:
        sport = next((s for s in ["bóng rổ", "tennis", "yoga", "cầu lông"] if s in q), None)

    return {
        "search_keywords": clean_q,
        "max_budget": max_budget,
        "category_id": cat_id,
        "brand_name": brand,
        "sport_type": sport,
        "gender": "nữ" if "nữ" in q else ("nam" if "nam" in q else "unisex"),
        "color_preference": color_pref,
        "price_tier": None,
        "excluded_keywords": excluded,
    }


def analyze_query(query_text: str) -> dict:
    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI that extracts shopping intents. ALWAYS return ONLY valid JSON. No explanations, no markdown code blocks."},
                {"role": "user", "content": _PROMPT_TEMPLATE.format(query=query_text)},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if not raw:
            logger.warning("⚠️ Empty response from LLM, falling back to regex")
            return _regex_fallback_extract(query_text)
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        intent = json.loads(raw)
        logger.info(f"🧠 Intent: {intent}")
        return intent
    except Exception as e:
        logger.error(f"❌ Together API error: {e}")
        return _regex_fallback_extract(query_text)
