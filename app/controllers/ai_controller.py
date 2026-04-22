from flask import Blueprint, jsonify, request
from app.services.rag_service import search_similar_products
import logging

logger = logging.getLogger(__name__)

# Khởi tạo Blueprint
ai_bp = Blueprint('ai_controller', __name__)

@ai_bp.route('/health', methods=['GET'])
def health_check():
    """API kiểm tra xem Python Service còn sống không"""
    return jsonify({"status": "OK", "message": "AI Service is running!"}), 200

@ai_bp.route('/search', methods=['POST'])
def smart_search():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Thiếu body JSON"}), 400

        keyword = data.get('query')
        category_id = data.get('category_id') # Lấy thêm filter từ frontend (nếu có)
        
        if not keyword:
            return jsonify({"error": "Thiếu từ khóa tìm kiếm (query)"}), 400

        logger.info(f"Nhận request search: query='{keyword}', category_id={category_id}")

        # Gọi Service để tìm top 5 sản phẩm, truyền thêm category_id nếu có
        product_ids = search_similar_products(keyword, target_category_id=category_id, top_k=5)
        
        return jsonify({
            "status": "success",
            "query": keyword, 
            "product_ids": product_ids
        }), 200
        
    except Exception as e:
        logger.error(f"Lỗi API /search: {e}", exc_info=True)
        # Giấu lỗi chi tiết khỏi response để đảm bảo bảo mật
        return jsonify({"status": "error", "message": "Đã có lỗi nội bộ xảy ra khi tìm kiếm!"}), 500