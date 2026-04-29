import os
import time
from sqlalchemy import create_engine, text
from together import Together
from dotenv import load_dotenv

load_dotenv()
DB_URL       = os.getenv("DATABASE_URL")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")

_MODEL = "intfloat/multilingual-e5-large-instruct"
# Số sản phẩm gửi embed 1 lần — Together AI cho phép batch lớn
_BATCH_SIZE = 20


def _build_content(row: dict) -> str:
    """
    Tạo chuỗi text đại diện cho sản phẩm.
    Nguyên tắc: nội dung phải giống ngôn ngữ tự nhiên mà người dùng hay tìm,
    tránh viết theo format máy móc.
    """
    parts: list[str] = []

    # Danh mục và thương hiệu đặt đầu — trọng số cao trong embedding
    if row["category_name"]:
        parts.append(f"Danh mục: {row['category_name']}.")
    if row["brand_name"]:
        parts.append(f"Thương hiệu: {row['brand_name']}.")

    parts.append(f"Sản phẩm: {row['name']}.")

    # Phân khúc giá (ngôn ngữ tự nhiên giúp khớp với "giá rẻ", "cao cấp")
    price = int(row["base_price"])
    if price < 500_000:
        tier_label = "giá rẻ, phổ thông"
    elif price < 1_500_000:
        tier_label = "tầm trung"
    else:
        tier_label = "cao cấp, chuyên nghiệp"
    parts.append(f"Giá: {price:,} VNĐ ({tier_label}).")

    if row["description"]:
        parts.append(f"Mô tả: {row['description']}.")

    # Màu sắc, kích cỡ, chất liệu — quan trọng cho filter phủ định
    if row["available_attributes"]:
        parts.append(f"Các phiên bản màu sắc và kích cỡ: {row['available_attributes']}.")

    # Giới tính — giúp khớp khi người dùng hỏi "nam" / "nữ"
    if row["gender_tags"]:
        parts.append(f"Phù hợp cho: {row['gender_tags']}.")

    # Môn thể thao phù hợp
    if row["sport_tags"]:
        parts.append(f"Phù hợp cho môn: {row['sport_tags']}.")

    return " ".join(parts)


def seed_database():
    print("🚀 Bắt đầu Vectorization với Together AI (multilingual-e5-large-instruct)...")

    client = Together(api_key=TOGETHER_KEY)
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE product_embeddings;"))
        conn.commit()

        # Query đầy đủ: gom màu sắc/size, thêm gender_tags và sport_tags từ ProductSpec
        query = text("""
            SELECT
                p.id,
                p.name,
                p.description,
                p.base_price,
                p.category_id,
                p.brand_id,
                c.name  AS category_name,
                b.name  AS brand_name,
                (
                    SELECT string_agg(DISTINCT av.value, ', ' ORDER BY av.value)
                    FROM product_skus sk
                    JOIN sku_values   sv ON sv.product_sku_id = sk.id
                    JOIN attribute_values av ON av.id = sv.attribute_value_id
                    WHERE sk.product_id = p.id
                ) AS available_attributes,
                -- Lấy giới tính từ ProductSpec (nếu có attribute tên 'Giới tính')
                (
                    SELECT ps.value
                    FROM product_specs ps
                    JOIN attributes a ON a.id = ps.attribute_id
                    WHERE ps.product_id = p.id
                      AND LOWER(a.name) IN ('giới tính', 'gender')
                    LIMIT 1
                ) AS gender_tags,
                -- Lấy môn thể thao từ ProductSpec (nếu có attribute tên 'Môn thể thao')
                (
                    SELECT ps.value
                    FROM product_specs ps
                    JOIN attributes a ON a.id = ps.attribute_id
                    WHERE ps.product_id = p.id
                      AND LOWER(a.name) IN ('môn thể thao', 'sport', 'hoạt động')
                    LIMIT 1
                ) AS sport_tags
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.id
            LEFT JOIN brands     b ON p.brand_id     = b.id
            WHERE p.is_active = TRUE
        """)

        rows = conn.execute(query).mappings().all()
        total = len(rows)
        print(f"📦 Tổng cộng {total} sản phẩm cần vectorize.")

        # Xử lý theo batch — Together AI hỗ trợ batch embed natively
        for batch_start in range(0, total, _BATCH_SIZE):
            batch = rows[batch_start: batch_start + _BATCH_SIZE]
            contents = [_build_content(dict(row)) for row in batch]

            # Gọi batch embed 1 lần cho cả batch — nhanh hơn gọi từng cái
            response = client.embeddings.create(model=_MODEL, input=contents)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            vectors = [item.embedding for item in sorted_data]

            for row, content, vector in zip(batch, contents, vectors):
                conn.execute(
                    text("""
                        INSERT INTO product_embeddings
                            (product_id, category_id, brand_id, content, embedding)
                        VALUES
                            (:pid, :cat_id, :brand_id, :content, :embedding)
                    """),
                    {
                        "pid":      row["id"],
                        "cat_id":   row["category_id"],
                        "brand_id": row["brand_id"],
                        "content":  content,
                        "embedding": str(vector),
                    },
                )

            conn.commit()
            print(f"  ✅ Đã xử lý {min(batch_start + _BATCH_SIZE, total)}/{total}")

            # Nghỉ nhỏ giữa các batch
            if batch_start + _BATCH_SIZE < total:
                time.sleep(0.5)

        print("🎉 Vectorization hoàn tất!")


if __name__ == "__main__":
    seed_database()