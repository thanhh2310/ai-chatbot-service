import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from app.utils.config import Config

logger = logging.getLogger(__name__)

class DatabaseService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseService, cls).__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        try:
            # Tối ưu connection pool
            self.engine = create_engine(
                Config.DATABASE_URL,
                pool_size=10,          # Giữ tối đa 10 connection sẵn sàng
                max_overflow=20,       # Cho phép vượt 20 connection khi tải cao
                pool_timeout=30,       # Chờ 30s nếu hết connection
                pool_recycle=1800,     # Reset connection sau 30 phút để tránh time-out từ DB
            )
            # Dùng scoped_session để an toàn trong môi trường đa luồng (multi-thread) của Flask
            self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
            logger.info("✅ Đã khởi tạo Connection Pool với Database")
        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo Database connection: {e}")
            raise

    def get_session(self):
        """Trả về một database session"""
        return self.SessionLocal()

    def get_engine(self):
        return self.engine

# Singleton instance để dùng chung trong toàn bộ app
db = DatabaseService()
