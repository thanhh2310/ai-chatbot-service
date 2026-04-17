from flask import Blueprint, jsonify, request
from app.services.rag_service import search_similar_products

# Khởi tạo Blueprint (Gom nhóm các API lại với nhau)
ai_bp = Blueprint('ai_controller', __name__)

@ai_bp.route('/health', methods=['GET'])
def health_check():
    """API dùng để Spring Boot kiểm tra xem Python còn sống không"""
    return jsonify({"status": "OK", "message": "AI Service is running!"}), 200

@ai_bp.route('/search', methods=['POST'])
def smart_search():
    try:
        data = request.json
        keyword = data.get('query')
        
        # Kiểm tra nếu Frontend/Java gửi thiếu dữ liệu
        if not keyword:
            return jsonify({"error": "Thiếu từ khóa tìm kiếm (query)"}), 400

        # Gọi Service để tìm top 5 sản phẩm
        product_ids = search_similar_products(keyword, top_k=5)
        
        return jsonify({
            "status": "success",
            "query": keyword, 
            "product_ids": product_ids
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500