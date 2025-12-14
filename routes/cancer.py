from flask import Blueprint, render_template, request
import joblib
import os
import numpy as np

# Blueprint 생성
cancer_bp = Blueprint(
    "cancer",
    __name__,
    url_prefix="/cancer"
)

# 경로 설정 (정석)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# 모델 로드
model = joblib.load(
    os.path.join(MODEL_DIR, "cancer_logistic.pkl")
)

# Breast Cancer target 이름
TARGET_NAMES = ["악성 (Malignant)", "양성 (Benign)"]

# 홈 화면
@cancer_bp.route("/", methods=["GET"])
def home():
    return render_template("cancer.html")

# 예측
@cancer_bp.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["mean_radius"]),
            float(request.form["mean_texture"]),
            float(request.form["mean_perimeter"]),
            float(request.form["mean_area"]),
            float(request.form["mean_smoothness"]),
            float(request.form["mean_compactness"]),
            float(request.form["mean_concavity"]),
            float(request.form["mean_concave_points"]),
            float(request.form["mean_symmetry"]),
            float(request.form["mean_fractal_dimension"]),
        ]

        X = np.array(features).reshape(1, -1)

        # 예측
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][pred]

        TARGET_NAMES = {
            0: "악성 (Malignant)",
            1: "양성 (Benign)"
        }

        return render_template(
            "cancer.html",
            prediction=TARGET_NAMES[pred],
            probability=round(proba * 100, 2),
            error=None
        )

    except Exception as e:
        print("🔥 예측 에러:", e)
        return render_template(
            "cancer.html",
            prediction=None,
            probability=None,
            error=str(e)
        )
