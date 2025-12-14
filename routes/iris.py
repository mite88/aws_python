from flask import Blueprint, render_template, request
import joblib
import os
from ml.iris_utils import CLASS_NAMES

# ✅ 1. Blueprint 먼저 생성
iris_bp = Blueprint('iris', __name__)

# 2. 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 3. 모델 로드
# 🔥 여러 모델 로드
MODELS = {
    "Logistic Regression": joblib.load(
        os.path.join(BASE_DIR, '..', 'models', 'iris_logistic.pkl')
    ),
    "Decision Tree": joblib.load(
        os.path.join(BASE_DIR, '..', 'models', 'iris_tree.pkl')
    ),
    "Random Forest": joblib.load(
        os.path.join(BASE_DIR, '..', 'models', 'iris_forest.pkl')
    )
}

# 4. 홈 페이지
@iris_bp.route('/', methods=['GET'])
def home():
    """
    Iris 예측 입력 화면
    """
    return render_template('index.html')

# 5. 예측 처리
@iris_bp.route('/iris/predict', methods=['POST'])
def predict():
    features = [[
        float(request.form['sepal_length']),
        float(request.form['sepal_width']),
        float(request.form['petal_length']),
        float(request.form['petal_width'])
    ]]

    results = []

    for model_name, model in MODELS.items():
        pred_idx = model.predict(features)[0]
        pred_label = CLASS_NAMES[pred_idx]

        proba = model.predict_proba(features)[0]
        proba_dict = {
            CLASS_NAMES[i]: round(float(p), 3)
            for i, p in enumerate(proba)
        }

        results.append({
            "model": model_name,
            "prediction": pred_label,
            "probabilities": proba_dict
        })

    return render_template(
        "index.html",
        results=results
    )
