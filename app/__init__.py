from flask import Flask

def create_app():
    # Khởi tạo app
    app = Flask(__name__)

    # Import các controllers (API Routes)
    from app.controllers.ai_controller import ai_bp
    
    # Đăng ký URL prefix (Giống @RequestMapping("/api/ai"))
    app.register_blueprint(ai_bp, url_prefix='/api/ai')

    # Khởi chạy quy trình Auto-Sync nhúng vector ngầm
    from app.services.sync_service import start_polling
    start_polling()

    return app