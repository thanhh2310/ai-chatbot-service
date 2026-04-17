from app import create_app
from app.utils.config import Config

app = create_app()

if __name__ == '__main__':
    print(f"🚀 AI Service đang chạy tại: http://localhost:{Config.PORT}")
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)