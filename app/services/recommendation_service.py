import logging
from sqlalchemy import text
from app.services.db_service import db
import numpy as np

engine = db.get_engine()

logger = logging.getLogger(__name__)

# Số seed product lấy từ mỗi nguồn (orders, wishlist)
_SEEDS_PER_SOURCE = 3
# Ứng viên lấy mỗi seed trước khi dedup — luôn lấy nhiều hơn limit
_CANDIDATES_PER_SEED = 10


def get_recommendations_for_user(user_id: int, limit: int = 6) -> list[int]:
    """
    Gợi ý sản phẩm cá nhân hóa dựa trên lịch sử mua và wishlist.

    Chiến lược:
      - Lấy tối đa 3 SP mua gần nhất + 3 SP wishlist gần nhất làm "seed"
      - Với mỗi seed: tìm top-N SP tương tự (loại trừ đã mua/đã thích/chính nó)
      - Gom kết quả, dedup, sắp xếp theo tần suất xuất hiện (popularity fusion)
      - Fallback: sản phẩm mới nhất nếu user chưa có dữ liệu
    """
    try:
        with engine.connect() as conn:
            # ── 1. Lấy danh sách SP đã mua và đã wishlist để loại trừ ──────
            bought_ids = _get_bought_product_ids(conn, user_id)
            wished_ids = _get_wished_product_ids(conn, user_id)
            exclude_ids = bought_ids | wished_ids  # union

            # ── 2. Lấy seed products ──────────────────────────────────────
            # Lấy nhiều seed hơn để vector trung bình đa dạng hơn
            seed_ids_from_orders   = _get_recent_product_ids(conn, "orders",    user_id, _SEEDS_PER_SOURCE)
            seed_ids_from_wishlist = _get_recent_product_ids(conn, "wishlists", user_id, _SEEDS_PER_SOURCE)
            seed_ids = list(dict.fromkeys(seed_ids_from_orders + seed_ids_from_wishlist))  # dedup, giữ thứ tự

            # ── 3. Fallback: user mới chưa có dữ liệu ────────────────────
            if not seed_ids:
                return _get_popular_products(conn, limit)

            # ── 4. Lấy embedding của các seed ────────────────────────────
            seed_vectors_dict = _get_embeddings(conn, seed_ids)
            if not seed_vectors_dict:
                return _get_popular_products(conn, limit)

            # ── 5. Tính Vector Trọng Tâm (Centroid User Profile) ─────────
            # Chuyển các string vector "[0.1, 0.2...]" thành mảng numpy floats
            vector_arrays = []
            for vec_str in seed_vectors_dict.values():
                # Xử lý string dạng "[0.123, -0.456, ...]" thành list float
                clean_str = vec_str.strip("[]")
                vec_float = [float(x) for x in clean_str.split(",")]
                vector_arrays.append(vec_float)
            
            # Tính trung bình cộng của tất cả các vector
            centroid_vector = np.mean(vector_arrays, axis=0)
            centroid_vector_str = f"[{','.join(map(str, centroid_vector))}]"

            # ── 6. Tìm SP tương tự với Vector Sở Thích (Chỉ 1 câu Query) ─
            candidates = _find_similar(
                conn=conn,
                seed_vector=centroid_vector_str,
                exclude_ids=exclude_ids, # Loại trừ đồ đã mua/thích
                top_k=limit,
            )
            
            result = [pid for pid, _ in candidates]
            logger.info(f"✅ Recommendations user={user_id}: {result}")
            return result

    except Exception as e:
        logger.error(f"❌ Recommendation lỗi user={user_id}: {e}", exc_info=True)
        return []


# ── PRIVATE HELPERS ──────────────────────────────────────────────────────────

def _get_bought_product_ids(conn, user_id: int) -> set[int]:
    """Toàn bộ SP user đã mua — dùng để loại trừ khỏi gợi ý."""
    rows = conn.execute(text("""
        SELECT DISTINCT ps.product_id
        FROM orders o
        JOIN order_items  oi ON o.id  = oi.order_id
        JOIN product_skus ps ON oi.product_sku_id = ps.id
        WHERE o.user_id = :uid
    """), {"uid": user_id}).mappings().all()
    return {r["product_id"] for r in rows}


def _get_wished_product_ids(conn, user_id: int) -> set[int]:
    """Toàn bộ SP trong wishlist — dùng để loại trừ khỏi gợi ý."""
    rows = conn.execute(text("""
        SELECT DISTINCT product_id FROM wishlists WHERE user_id = :uid
    """), {"uid": user_id}).mappings().all()
    return {r["product_id"] for r in rows}


def _get_recent_product_ids(conn, source: str, user_id: int, limit: int) -> list[int]:
    """
    Lấy product_id gần nhất từ orders hoặc wishlists.
    Trả về nhiều seed hơn 1 để tăng độ đa dạng gợi ý.
    """
    if source == "orders":
        sql = text("""
            SELECT ps.product_id
            FROM orders o
            JOIN order_items  oi ON o.id  = oi.order_id
            JOIN product_skus ps ON oi.product_sku_id = ps.id
            WHERE o.user_id = :uid
            GROUP BY ps.product_id
            ORDER BY MAX(o.created_at) DESC
            LIMIT :limit
        """)
    else:  # wishlists
        sql = text("""
            SELECT product_id FROM wishlists
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT :limit
        """)
    rows = conn.execute(sql, {"uid": user_id, "limit": limit}).mappings().all()
    return [r["product_id"] for r in rows]


def _get_embeddings(conn, product_ids: list[int]) -> dict[int, str]:
    """Lấy embedding của nhiều sản phẩm trong 1 query."""
    if not product_ids:
        return {}
    rows = conn.execute(text("""
        SELECT product_id, embedding
        FROM product_embeddings
        WHERE product_id = ANY(:pids)
    """), {"pids": product_ids}).mappings().all()
    return {r["product_id"]: r["embedding"] for r in rows}


def _find_similar(conn, seed_vector: str, exclude_ids: set[int], top_k: int) -> list[tuple[int, float]]:
    """
    Tìm SP tương tự với 1 seed vector.
    Loại trừ các SP trong exclude_ids ngay trong SQL — tránh subquery động.
    """
    if not exclude_ids:
        exclude_ids = {-1}  # tránh IN () rỗng gây lỗi SQL

    rows = conn.execute(text("""
        SELECT DISTINCT pe.product_id, MIN(pe.embedding <=> CAST(:vector AS vector)) AS distance
        FROM product_embeddings pe
        JOIN products p ON p.id = pe.product_id
        WHERE p.is_active = TRUE
          AND pe.product_id != ALL(:exclude_ids)
        GROUP BY pe.product_id
        ORDER BY distance ASC
        LIMIT :top_k
    """), {
        "vector":      seed_vector,
        "exclude_ids": list(exclude_ids),
        "top_k":       top_k,
    }).mappings().all()

    return [(r["product_id"], float(r["distance"])) for r in rows]


def _get_popular_products(conn, limit: int) -> list[int]:
    """Fallback cho user mới: trả về SP được mua nhiều nhất đang active."""
    rows = conn.execute(text("""
        SELECT ps.product_id, SUM(oi.quantity) AS total_sold
        FROM orders o
        JOIN order_items  oi ON o.id  = oi.order_id
        JOIN product_skus ps ON oi.product_sku_id = ps.id
        JOIN products p ON p.id = ps.product_id
        WHERE o.order_status NOT IN ('CANCELLED')
          AND p.is_active = TRUE
        GROUP BY ps.product_id
        ORDER BY total_sold DESC
        LIMIT :limit
    """), {"limit": limit}).mappings().all()
    return [r["product_id"] for r in rows]


