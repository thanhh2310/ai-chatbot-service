"""
Script tạo dữ liệu test cho Recommendation API.
Chạy: source venv/bin/activate && python seed_recommendation_data.py
"""
import random
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://vutienthanh:@localhost:5432/datn")
engine = create_engine(DATABASE_URL)

# Số chiều embedding (Together AI multilingual-e5-large-instruct)
EMBED_DIM = 1024

def random_vector(dim=EMBED_DIM):
    """Tạo vector ngẫu nhiên (giả lập embedding)."""
    return [random.gauss(0, 0.1) for _ in range(dim)]

def run():
    with engine.connect() as conn:
        # ── 1. Brands ─────────────────────────────────────────────────────
        brands = [
            ("Nike", "nike"),
            ("Adidas", "adidas"),
            ("Puma", "puma"),
            ("Under Armour", "under-armour"),
            ("Li-Ning", "li-ning"),
            ("New Balance", "new-balance"),
        ]
        for name, slug in brands:
            conn.execute(text("""
                INSERT INTO brands (name, slug, is_active)
                VALUES (:name, :slug, TRUE)
                ON CONFLICT (slug) DO NOTHING
            """), {"name": name, "slug": slug})
        conn.commit()

        # Build brand slug -> id mapping (existing + newly created)
        brand_rows = conn.execute(text("SELECT id, slug FROM brands")).mappings().all()
        brand_map = {r["slug"]: r["id"] for r in brand_rows}
        print(f"📦 Brands: {brand_map}")

        # ── 2. Categories ─────────────────────────────────────────────────
        categories = [
            ("Giày/Dép", "giay-dep"),
            ("Quần Áo", "quan-ao"),
            ("Phụ Kiện", "phu-kien"),
        ]
        for name, slug in categories:
            conn.execute(text("""
                INSERT INTO categories (name, slug, is_active)
                VALUES (:name, :slug, TRUE)
                ON CONFLICT (slug) DO NOTHING
            """), {"name": name, "slug": slug})
        conn.commit()

        cat_rows = conn.execute(text("SELECT id, slug FROM categories")).mappings().all()
        cat_map = {r["slug"]: r["id"] for r in cat_rows}
        print(f"📂 Categories: {cat_map}")

        # ── 3. Products ───────────────────────────────────────────────────
        # Use brand slugs and category slugs instead of hardcoded IDs
        products = [
            # Giày - Nike
            ("Giày Nike Air Max 270", "giay-nike-air-max-270", 2500000, "giay-dep", "nike", "Giày chạy bộ Nike Air Max 270 nam nữ, đế êm ái, thoáng khí, phù hợp chạy bộ và gym."),
            ("Giày Nike React Infinity Run", "giay-nike-react-infinity", 3200000, "giay-dep", "nike", "Giày chạy bộ chuyên nghiệp Nike React, đệm êm, hỗ trợ chân tốt, màu đen trắng."),
            ("Giày Nike ZoomX Vaporfly", "giay-nike-zoomx-vaporfly", 4500000, "giay-dep", "nike", "Giày marathon cao cấp Nike ZoomX, siêu nhẹ, đàn hồi tốt, dành cho vận động viên."),
            ("Giày Nike Pegasus 40", "giay-nike-pegasus-40", 2800000, "giay-dep", "nike", "Giày chạy bộ hàng ngày Nike Pegasus, bền bỉ, thoải mái, phù hợp mọi địa hình."),
            ("Giày Nike Free Run 5.0", "giay-nike-free-run", 2100000, "giay-dep", "nike", "Giày chạy bộ linh hoạt Nike Free Run, đế mỏng, cảm giác chân trần, nam nữ đều dùng."),

            # Giày - Adidas
            ("Giày Adidas Ultraboost 23", "giay-adidas-ultraboost-23", 3800000, "giay-dep", "adidas", "Giày chạy bộ Adidas Ultraboost, đệm Boost êm ái, thoáng khí, phù hợp chạy bộ đường dài."),
            ("Giày Adidas Supernova Eterno", "giay-adidas-supernova", 2900000, "giay-dep", "adidas", "Giày chạy bộ Adidas Supernova, ổn định, hỗ trợ tốt, giá tầm trung."),
            ("Giày Adidas Predator Edge", "giay-adidas-predator", 3500000, "giay-dep", "adidas", "Giày bóng đá Adidas Predator, bám sân tốt, kiểm soát bóng chính xác."),

            # Giày - Puma
            ("Giày Puma Velocity Nitro 3", "giay-puma-velocity-nitro", 2600000, "giay-dep", "puma", "Giày chạy bộ Puma Velocity Nitro, đệm Nitro nhẹ, đàn hồi cao, phù hợp chạy bộ hàng ngày."),
            ("Giày Puma Future Z Football", "giay-puma-future-z", 2400000, "giay-dep", "puma", "Giày bóng đá Puma Future Z, linh hoạt, bám sân, màu xanh neon."),

            # Quần Áo - Nike
            ("Áo thun Nike Dri-FIT", "ao-nike-drifit", 850000, "quan-ao", "nike", "Áo thun tập gym Nike Dri-FIT, thấm hút mồ hôi, thoáng mát, nam nữ."),
            ("Quần short Nike Flex", "quan-short-nike-flex", 650000, "quan-ao", "nike", "Quần short tập gym Nike Flex, co giãn tốt, nhẹ, phù hợp chạy bộ và gym."),
            ("Áo khoác Nike Windrunner", "ao-khoac-nike-windrunner", 1800000, "quan-ao", "nike", "Áo khoác gió Nike Windrunner, chống nước nhẹ, phù hợp chạy bộ ngoài trời."),

            # Quần Áo - Adidas
            ("Áo thun Adidas Aeroready", "ao-adidas-aeroready", 750000, "quan-ao", "adidas", "Áo thun tập luyện Adidas Aeroready, thoáng khí, thấm hút mồ hôi."),
            ("Quần dài Adidas Tiro", "quan-adidas-tiro", 950000, "quan-ao", "adidas", "Quần dài tập bóng đá Adidas Tiro, co giãn, thoải mái vận động."),

            # Quần Áo - Under Armour
            ("Áo compression Under Armour", "ao-ua-compression", 1200000, "quan-ao", "under-armour", "Áo bó sát Under Armour, compression, hỗ trợ cơ bắp, phù hợp gym và chạy bộ."),
            ("Quần short UA HeatGear", "quan-ua-heatgear", 850000, "quan-ao", "under-armour", "Quần short tập luyện UA HeatGear, mát mẻ, thoáng khí."),

            # Phụ Kiện - Nike
            ("Túi Nike Brasilia", "tui-nike-brasilia", 950000, "phu-kien", "nike", "Túi đeo vai Nike Brasilia, dung tích lớn, phù hợp đi tập gym và du lịch."),
            ("Mũ Nike Heritage", "mu-nike-heritage", 450000, "phu-kien", "nike", "Mũ lưỡi trai Nike Heritage, chống nắng, phong cách thể thao."),

            # Phụ Kiện - Li-Ning
            ("Vợt cầu lông Li-Ning", "vot-cau-long-li-ning", 1500000, "phu-kien", "li-ning", "Vợt cầu lông Li-Ning chuyên nghiệp, siêu nhẹ, đàn hồi tốt."),
            ("Giày cầu lông Li-Ning", "giay-cau-long-li-ning", 1800000, "giay-dep", "li-ning", "Giày cầu lông Li-Ning, bám sân tốt, bảo vệ cổ chân, phù hợp thi đấu."),

            # Giày - New Balance
            ("Giày New Balance Fresh Foam", "giay-nb-fresh-foam", 3100000, "giay-dep", "new-balance", "Giày chạy bộ New Balance Fresh Foam, đệm êm ái, thoải mái, phù hợp chạy bộ đường dài."),
            ("Giày New Balance 574", "giay-nb-574", 2200000, "giay-dep", "new-balance", "Giày sneaker New Balance 574 cổ điển, phong cách casual, nam nữ."),
        ]

        product_ids = []
        for name, slug, price, cat_slug, brand_slug, desc in products:
            cat_id = cat_map.get(cat_slug)
            brand_id = brand_map.get(brand_slug)
            if not cat_id or not brand_id:
                print(f"⚠️ Skipping '{name}': missing category '{cat_slug}' or brand '{brand_slug}'")
                continue
            result = conn.execute(text("""
                INSERT INTO products (name, slug, description, base_price, category_id, brand_id, is_active, created_at)
                VALUES (:name, :slug, :desc, :price, :cat_id, :brand_id, TRUE, NOW())
                ON CONFLICT (slug) DO NOTHING
                RETURNING id
            """), {"name": name, "slug": slug, "desc": desc, "price": price, "cat_id": cat_id, "brand_id": brand_id})
            pid = result.scalar()
            if pid:
                product_ids.append(pid)

        # If products already existed (ON CONFLICT DO NOTHING), fetch their IDs
        if len(product_ids) < len(products):
            for name, slug, price, cat_slug, brand_slug, desc in products:
                existing = conn.execute(text("SELECT id FROM products WHERE slug = :slug"), {"slug": slug}).first()
                if existing and existing[0] not in product_ids:
                    product_ids.append(existing[0])

        product_ids.sort()
        conn.commit()
        print(f"🏷️ Products created/found: {len(product_ids)} (IDs: {product_ids})")

        # ── 4. Product SKUs (only for newly created products) ─────────────
        sku_counter = 1
        sku_id_map = {}  # product_id -> list of sku_ids
        for pid in product_ids:
            existing_skus = conn.execute(text("SELECT id FROM product_skus WHERE product_id = :pid"), {"pid": pid}).fetchall()
            if existing_skus:
                sku_id_map[pid] = [r[0] for r in existing_skus]
                continue
            skus = [
                (f"SKU-{sku_counter:04d}", 0, 50),
                (f"SKU-{sku_counter+1:04d}", 0, 30),
            ]
            sku_counter += 2
            sku_ids = []
            for code, price_offset, stock in skus:
                product = conn.execute(text("SELECT base_price FROM products WHERE id = :pid"), {"pid": pid}).scalar()
                price = int(product) + price_offset
                result = conn.execute(text("""
                    INSERT INTO product_skus (product_id, sku_code, price, stock_quantity, is_active, created_at)
                    VALUES (:pid, :code, :price, :stock, TRUE, NOW())
                    RETURNING id
                """), {"pid": pid, "code": code, "price": price, "stock": stock})
                sku_id = result.scalar()
                sku_ids.append(sku_id)
            sku_id_map[pid] = sku_ids
        conn.commit()

        # ── 5. Product Embeddings ─────────────────────────────────────────
        # Skip if already exists; background sync_service will create real ones
        for pid in product_ids:
            existing = conn.execute(text("SELECT product_id FROM product_embeddings WHERE product_id = :pid"), {"pid": pid}).first()
            if existing:
                continue
            conn.execute(text("""
                INSERT INTO product_embeddings (product_id, category_id, brand_id, content, embedding)
                VALUES (:pid, :cat_id, :brand_id, :content, :embedding)
            """), {
                "pid": pid,
                "cat_id": conn.execute(text("SELECT category_id FROM products WHERE id = :pid"), {"pid": pid}).scalar(),
                "brand_id": conn.execute(text("SELECT brand_id FROM products WHERE id = :pid"), {"pid": pid}).scalar(),
                "content": f"San pham ID {pid}",
                "embedding": str(random_vector()),
            })
        conn.commit()

        # ── 6. Users ──────────────────────────────────────────────────────
        existing_user = conn.execute(text("SELECT id FROM users WHERE id = 1")).first()
        if not existing_user:
            conn.execute(text("""
                INSERT INTO users (id, email, password_hash, first_name, last_name, is_active, created_at)
                VALUES (1, 'test@example.com', 'hashed_password', 'Test', 'User', TRUE, NOW())
            """))
            conn.commit()

        # ── 7. Orders + Order Items ───────────────────────────────────────
        bought_products = product_ids[:3] if len(product_ids) >= 3 else product_ids
        order_total = sum([
            conn.execute(text("SELECT base_price FROM products WHERE id = :pid"), {"pid": pid}).scalar()
            for pid in bought_products
        ])

        result = conn.execute(text("""
            INSERT INTO orders (user_id, total_amount, order_status, payment_status,
                                shipping_address, shipping_city, subtotal, shipping_fee, discount_amount,
                                payment_method_id, created_at)
            VALUES (1, :total, 'DELIVERED', 'PAID', '123 Test St', 'Hanoi', :total, 30000, 0, 1, NOW() - INTERVAL '7 days')
            RETURNING id
        """), {"total": order_total})
        order_id = result.scalar()

        for pid in bought_products:
            sku_id = sku_id_map[pid][0]
            price = conn.execute(text("SELECT price FROM product_skus WHERE id = :sid"), {"sid": sku_id}).scalar()
            conn.execute(text("""
                INSERT INTO order_items (order_id, product_sku_id, quantity, price, discount)
                VALUES (:oid, :sku_id, 1, :price, 0)
            """), {"oid": order_id, "sku_id": sku_id, "price": price})
        conn.commit()

        # ── 8. Wishlist ───────────────────────────────────────────────────
        wished_products = product_ids[3:5] if len(product_ids) >= 5 else product_ids[3:]
        for pid in wished_products:
            conn.execute(text("""
                INSERT INTO wishlists (user_id, product_id, created_at)
                VALUES (1, :pid, NOW() - INTERVAL '3 days')
                ON CONFLICT (user_id, product_id) DO NOTHING
            """), {"pid": pid})
        conn.commit()

        # ── Summary ───────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("✅ Done! Test data summary:")
        print(f"   - Products: {product_ids}")
        print(f"   - User ID=1 bought product IDs: {bought_products}")
        print(f"   - User ID=1 wished product IDs: {wished_products}")
        print()
        print("🧪 Test recommendation API:")
        print("   curl 'http://localhost:5001/api/ai/recommendations?user_id=1'")
        print("=" * 60)

if __name__ == "__main__":
    run()
