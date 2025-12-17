# %%
from flask import Blueprint, render_template, request
import joblib
import os
import numpy as np

wine_bp = Blueprint("wine", __name__)

# 모델 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "wine_logistic.pkl")

model = joblib.load(MODEL_PATH)

# 표시용 클래스 이름(원하면 바꿔도 됨)
TARGET_NAMES = ["Class 0", "Class 1", "Class 2"]

# Wine feature names (scikit-learn 기본 순서와 동일)
FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

@wine_bp.route("/", methods=["GET"])
def home():
    return render_template("wine.html", features=FEATURES)

@wine_bp.route("/predict", methods=["POST"])
def predict():
    try:
        # 폼에서 입력받은 값들을 FEATURES 순서대로 꺼내기
        values = []
        for f in FEATURES:
            v = float(request.form.get(f, "").strip())
            values.append(v)

        X_input = np.array(values).reshape(1, -1)

        pred = int(model.predict(X_input)[0])
        proba = model.predict_proba(X_input)[0]  # [p0, p1, p2]

        result = {
            "pred_idx": pred,
            "pred_name": TARGET_NAMES[pred],
            "proba": [float(p) for p in proba],
        }

        return render_template("wine.html", features=FEATURES, result=result, form=request.form)

    except Exception as e:
        return render_template("wine.html", features=FEATURES, error=str(e), form=request.form)

# %%
