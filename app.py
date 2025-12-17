# 서버용
# app.py
from flask import Flask
from routes.iris import iris_bp
from routes.cancer import cancer_bp
from routes.wine_routes import wine_bp
from routes.main import main_bp   # 메인 허브용


app = Flask(__name__)


# Blueprint 등록
app.register_blueprint(main_bp)            
app.register_blueprint(iris_bp, url_prefix='/iris')
app.register_blueprint(cancer_bp)
app.register_blueprint(wine_bp, url_prefix="/wine")



if __name__ == '__main__':
    app.run(debug=True)