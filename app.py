# 서버용
# app.py
from flask import Flask
from routes.iris import iris_bp


app = Flask(__name__)


# Blueprint 등록
app.register_blueprint(iris_bp)


if __name__ == '__main__':
    app.run(debug=True)