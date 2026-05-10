from flask import Flask
from flask_cors import CORS

def create_app():
    # Khởi tạo app
    app = Flask(__name__)

    # Cho phép mọi origin (bao gồm file://) để test HTML local
    CORS(app, origins="*", supports_credentials=False)

    # Import các controllers (API Routes)
    from app.controllers.ai_controller import ai_bp
    
    # Đăng ký URL prefix (Giống @RequestMapping("/api/ai"))
    app.register_blueprint(ai_bp, url_prefix='/api/ai')

    # Khởi chạy quy trình Auto-Sync nhúng vector ngầm
    from app.services.sync_service import start_polling
    start_polling()

    return app