# File: run.py
from flask import Flask, jsonify

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Giống @GetMapping("/") trong Spring Boot
@app.route('/', methods=['GET'])
def hello_world():
    # Spring Boot tự map Object ra JSON, Flask thì dùng hàm jsonify
    return jsonify({"message": "Hello World từ AI Service bằng Flask!"})

# Giống hàm public static void main(String[] args)
if __name__ == '__main__':
    # debug=True để khi bạn sửa code, server tự động restart (như devtools của Spring)
    # port=5000 là cổng mặc định của Flask (để không đụng port 8080 của Java)
    app.run(debug=True, port=5000)