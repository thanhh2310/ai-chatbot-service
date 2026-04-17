# File: app/services/rag_service.py
from sqlalchemy import create_engine, text
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.utils.config import Config
from sqlalchemy import text

# 1. Khởi tạo model nhúng (Giống hệt file seed_data.py)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    google_api_key=Config.GEMINI_API_KEY
)

# 2. Khởi tạo Engine kết nối Database
engine = create_engine(Config.DATABASE_URL)

def search_similar_products(query_text, target_category_id=None, top_k=5):
    try:
        # 1. Nhúng câu query của khách
        query_vector = embeddings_model.embed_query(query_text)
        
        # 2. Xây dựng câu SQL động
        if target_category_id:
            # HYBRID SEARCH: Lọc cứng category trước, so sánh vector sau
            sql = text("""
                SELECT product_id, content, (embedding <=> :vector) as distance
                FROM product_embeddings
                WHERE category_id = :cat_id
                ORDER BY embedding <=> :vector
                LIMIT :limit
            """)
            params = {
                "vector": str(query_vector), 
                "limit": top_k, 
                "cat_id": target_category_id
            }
        else:
            # VECTOR SEARCH THUẦN: Quét toàn bộ (Như cũ)
            sql = text("""
                SELECT product_id, content, (embedding <=> :vector) as distance
                FROM product_embeddings
                ORDER BY embedding <=> :vector
                LIMIT :limit
            """)
            params = {
                "vector": str(query_vector), 
                "limit": top_k
            }
            
        with engine.connect() as conn:
            results = conn.execute(sql, params).mappings().all()
            
            product_ids = [row['product_id'] for row in results]
            
            print(f"🔍 Khách tìm: '{query_text}' | Category Filter: {target_category_id}")
            for row in results:
                print(f" -> ID {row['product_id']} | Độ lệch: {row['distance']:.4f} | {row['content'][:50]}...")
                
            return product_ids
            
    except Exception as e:
        print(f"❌ Lỗi khi search vector: {e}")
        return []