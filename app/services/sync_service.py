import threading
import time
import logging
from app.services.db_service import db
from sqlalchemy import text
from app.services.embedding_service import embed_long_texts

logger = logging.getLogger(__name__)

# Together AI cho phép batch lớn hơn Gemini — dùng batch 20 cho hiệu quả
_BATCH_SIZE = 20

def _build_content(row: dict) -> str:
    """Tạo chuỗi text đại diện cho sản phẩm."""
    parts: list[str] = []
    if row["category_name"]: parts.append(f"Danh mục: {row['category_name']}.")
    if row["brand_name"]: parts.append(f"Thương hiệu: {row['brand_name']}.")
    parts.append(f"Sản phẩm: {row['name']}.")
    
    price = int(row["base_price"])
    if price < 500_000: tier_label = "giá rẻ, phổ thông"
    elif price < 1_500_000: tier_label = "tầm trung"
    else: tier_label = "cao cấp, chuyên nghiệp"
    parts.append(f"Giá: {price:,} VNĐ ({tier_label}).")
    
    if row["description"]: parts.append(f"Mô tả: {row['description']}.")
    if row["available_attributes"]: parts.append(f"Các phiên bản màu sắc và kích cỡ: {row['available_attributes']}.")
    if row["gender_tags"]: parts.append(f"Phù hợp cho: {row['gender_tags']}.")
    if row["sport_tags"]: parts.append(f"Phù hợp cho môn: {row['sport_tags']}.")
    return " ".join(parts)


def sync_new_products():
    session = db.get_session()
    try:
        query = text("""
            SELECT
                p.id, p.name, p.description, p.base_price, p.category_id, p.brand_id,
                c.name AS category_name, b.name AS brand_name,
                (
                    SELECT string_agg(DISTINCT av.value, ', ' ORDER BY av.value)
                    FROM product_skus sk
                    JOIN sku_values sv ON sv.product_sku_id = sk.id
                    JOIN attribute_values av ON av.id = sv.attribute_value_id
                    WHERE sk.product_id = p.id
                ) AS available_attributes,
                (
                    SELECT av.value 
                    FROM product_specs ps
                    JOIN attribute_values av ON av.id = ps.attribute_value_id
                    JOIN attributes a ON a.id = av.attribute_id
                    WHERE ps.product_id = p.id AND LOWER(a.name) IN ('giới tính', 'gender')
                    LIMIT 1
                ) AS gender_tags,
                (
                    SELECT av.value 
                    FROM product_specs ps
                    JOIN attribute_values av ON av.id = ps.attribute_value_id
                    JOIN attributes a ON a.id = av.attribute_id
                    WHERE ps.product_id = p.id AND LOWER(a.name) IN ('môn thể thao', 'sport', 'hoạt động')
                    LIMIT 1
                ) AS sport_tags
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN brands b ON p.brand_id = b.id
            LEFT JOIN product_embeddings pe ON pe.product_id = p.id
            WHERE p.is_active = TRUE AND pe.product_id IS NULL
        """)
        
        rows = session.execute(query).mappings().all()
        if not rows: return
        
        logger.info(f"🔄 Phát hiện {len(rows)} sản phẩm mới chưa vectorize. Bắt đầu xử lý...")
        
        contents = [_build_content(dict(row)) for row in rows]
        
        # Text chunking: mỗi product content có thể vượt 512 tokens của embedding model.
        # embed_long_texts sẽ chunk, embed từng chunk, rồi gộp thành 1 vector/product.
        vectors = embed_long_texts(contents, batch_size=_BATCH_SIZE)
        
        for row, content, vector in zip(rows, contents, vectors):
            session.execute(text("""
                INSERT INTO product_embeddings (product_id, category_id, brand_id, content, embedding)
                VALUES (:pid, :cat_id, :brand_id, :content, :embedding)
            """), {
                "pid": row["id"], "cat_id": row["category_id"], "brand_id": row["brand_id"],
                "content": content, "embedding": str(vector)
            })
            
        session.commit()
        logger.info(f"✅ Auto-Sync hoàn tất vectorize {len(rows)} sản phẩm mới siêu tốc.")
        
    except Exception as e:
        logger.error(f"❌ Lỗi Auto-Sync Vector: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()

def start_polling():
    def poll():
        logger.info("🕒 Background Sync Thread đã chạy ngầm...")
        while True:
            try:
                sync_new_products()
            except Exception as e:
                logger.error(f"Polling loop error: {e}")
            time.sleep(15) # Check DB mỗi 15 giây
            
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
