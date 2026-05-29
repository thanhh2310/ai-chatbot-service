# Sử dụng Python 3.11 phiên bản slim (nhẹ, nhưng vẫn đủ thư viện C chuẩn để chạy psycopg2 và numpy)
FROM python:3.11-slim
WORKDIR /app

# Thiết lập biến môi trường giúp Python chạy ổn định trong Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy file requirements.txt
COPY requirements.txt .

# Cài đặt thư viện AI từ requirements.txt VÀ cài bổ sung gunicorn để chạy production
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy toàn bộ mã nguồn Flask AI vào container
COPY . .

# Mở cổng 5000 để giao tiếp trong mạng Docker
EXPOSE 5000

# Khởi chạy bằng Gunicorn
# - --bind 0.0.0.0:5000: Lắng nghe mọi kết nối trên cổng 5000
# - --workers 2: Tạo 2 tiến trình xử lý song song. Do các model AI khá tốn RAM, không nên set quá cao.
# - --timeout 120: AI có thể mất thời gian để trả lời, set timeout cao (120s) để tránh lỗi Timeout.
# - app:app: (Tên file python chứa ứng dụng của bạn) : (Tên biến khởi tạo Flask app)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120", "app:create_app()"]