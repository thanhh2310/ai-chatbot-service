import os
from sqlalchemy import create_engine, text
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load cấu hình
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def seed_database():
    print("🚀 Bắt đầu quá trình Vectorization...")

    # 1. Khởi tạo công cụ nhúng (Embeddings)
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", 
        google_api_key=GEMINI_KEY
    )

    # 2. Kết nối Database
    engine = create_engine(DB_URL)
    
    with engine.connect() as conn:
        # Xóa dữ liệu cũ trong bảng product_embeddings trước khi nhúng lại (tùy chọn)
        conn.execute(text("TRUNCATE TABLE product_embeddings;"))
        conn.commit()
        print("🗑️ Đã dọn sạch bảng product_embeddings cũ.")

        # 3. Kéo thêm category_id, brand_id và tên Brand để làm content xịn hơn
        query = """
            SELECT 
                p.id, p.name, p.description, p.base_price,
                p.category_id, p.brand_id,
                c.name as category_name,
                b.name as brand_name
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN brands b ON p.brand_id = b.id;
        """
        result = conn.execute(text(query)).mappings().all()

        if not result:
            print("❌ Bảng products của bạn đang trống! Hãy seed dữ liệu trước nhé.")
            return

        for row in result:
            product_id = row['id']
            cat_id = row['category_id']
            brand_id = row['brand_id']
            
            # 4. Gom thông tin thành 1 đoạn văn chuẩn mực (Prompt Engineering)
            content = f"Danh mục: {row['category_name']}. "
            if row['brand_name']:
                content += f"Thương hiệu: {row['brand_name']}. "
            content += f"Sản phẩm: {row['name']}. "
            content += f"Giá: {int(row['base_price'])} VNĐ. "
            if row['description']:
                content += f"Mô tả: {row['description']}"
            
            print(f"Đang xử lý SP ID {product_id}...")

            # 5. Gọi API Gemini để biến đoạn văn thành Vector mảng số (768 chiều)
            vector = embeddings_model.embed_query(content)

            # 6. LƯU Ý SỬA INSERT: Thêm category_id và brand_id vào
            insert_query = text("""
                INSERT INTO product_embeddings (product_id, category_id, brand_id, content, embedding) 
                VALUES (:pid, :cat_id, :brand_id, :content, :embedding)
            """)
            
            conn.execute(insert_query, {
                "pid": product_id, 
                "cat_id": cat_id,
                "brand_id": brand_id,
                "content": content, 
                "embedding": str(vector) # Ép kiểu về string cho pgvector
            })
            conn.commit()
            
        print("🎉 Quá trình nhúng toàn bộ sản phẩm đã hoàn tất thành công!")

if __name__ == "__main__":
    seed_database()