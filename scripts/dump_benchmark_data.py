#!/usr/bin/env python3
"""
Dump product/category/user data from DB for building benchmark test cases.
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # 1) All active products with category
    products = conn.execute(text("""
        SELECT p.id, p.name, p.base_price, p.is_active,
               c.name AS category_name, b.name AS brand_name
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        LEFT JOIN brands b ON b.id = p.brand_id
        WHERE p.is_active = TRUE
        ORDER BY p.id
    """)).mappings().all()
    
    print("=== PRODUCTS ===")
    for p in products:
        print(f"  ID {p['id']:3d}: {p['name']} | Cat: {p['category_name']} | Brand: {p['brand_name']} | Price: {p['base_price']}")
    print(f"  Total active products: {len(products)}")
    
    # 2) Categories
    categories = conn.execute(text("""
        SELECT c.id, c.name, COUNT(p.id) AS product_count
        FROM categories c
        LEFT JOIN products p ON p.category_id = c.id AND p.is_active = TRUE
        GROUP BY c.id, c.name
        ORDER BY c.id
    """)).mappings().all()
    
    print("\n=== CATEGORIES ===")
    for c in categories:
        print(f"  ID {c['id']:3d}: {c['name']} ({c['product_count']} products)")
    
    # 3) Users with interactions
    users = conn.execute(text("""
        SELECT u.id, u.height, u.weight,
               (SELECT COUNT(*) FROM user_interactions ui WHERE ui.user_id = u.id) AS interaction_count,
               (SELECT COUNT(*) FROM wishlists w WHERE w.user_id = u.id) AS wishlist_count,
               (SELECT COUNT(*) FROM reviews r WHERE r.user_id = u.id) AS review_count
        FROM users u
        ORDER BY u.id
        LIMIT 20
    """)).mappings().all()
    
    print("\n=== USERS (top 20) ===")
    for u in users:
        print(f"  User {u['id']:3d}: h={u['height']}cm w={u['weight']}kg | interactions={u['interaction_count']} wishlists={u['wishlist_count']} reviews={u['review_count']}")
    
    # 4) Available SKU attributes (size/color combos in stock)
    sku_attrs = conn.execute(text("""
        SELECT p.id AS product_id, p.name AS product_name,
               STRING_AGG(DISTINCT av.value, ', ' ORDER BY av.value) AS available_values,
               a.name AS attribute_name
        FROM products p
        JOIN product_skus ps ON ps.product_id = p.id AND ps.is_active = TRUE AND ps.stock_quantity > 0
        JOIN product_sku_attributes psa ON psa.product_sku_id = ps.id
        JOIN attribute_values av ON av.id = psa.attribute_value_id
        JOIN attributes a ON a.id = av.attribute_id
        WHERE p.is_active = TRUE
        GROUP BY p.id, p.name, a.name
        ORDER BY p.id, a.name
        LIMIT 80
    """)).mappings().all()
    
    print("\n=== SKU ATTRIBUTES (in stock) ===")
    for s in sku_attrs:
        print(f"  Product {s['product_id']:3d} ({s['product_name']}) | {s['attribute_name']}: {s['available_values']}")

    # 5) Brands
    brands = conn.execute(text("""
        SELECT b.id, b.name, COUNT(p.id) AS product_count
        FROM brands b
        LEFT JOIN products p ON p.brand_id = b.id AND p.is_active = TRUE
        GROUP BY b.id, b.name
        HAVING COUNT(p.id) > 0
        ORDER BY b.id
    """)).mappings().all()
    
    print("\n=== BRANDS ===")
    for b in brands:
        print(f"  ID {b['id']:3d}: {b['name']} ({b['product_count']} products)")
