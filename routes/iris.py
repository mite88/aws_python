from flask import Blueprint, render_template, request
import joblib
import os
from ml.iris_utils import CLASS_NAMES

# ✅ 1. Blueprint 먼저 생성
iris_bp = Blueprint('iris', __name__)

# 2. 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'iris_model.pkl')

# 3. 모델 로드
model = joblib.load(MODEL_PATH)

# 4. 홈 페이지
@iris_bp.route('/', methods=['GET'])
def home():
    """
    Iris 예측 입력 화면
    """
    return render_template('index.html')

# 5. 예측 처리
@iris_bp.route('/predict', methods=['POST'])
def predict():
    """
    HTML Form 입력 → 모델 예측 → 결과 반환
    """

    sepal_length = float(request.form['sepal_length'])
    sepal_width  = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width  = float(request.form['petal_width'])

    features = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    pred_idx = model.predict(features)[0]
    pred_label = CLASS_NAMES[pred_idx]

    return render_template(
        'index.html',
        prediction=pred_label
    )
