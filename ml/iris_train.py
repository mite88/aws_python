"""
Iris 데이터셋
- 여러 분류 모델을 학습
- 각 모델을 개별 pkl 파일로 저장
- Flask에서 동시에 불러와 비교하기 위한 학습 스크립트
"""

import os
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================
# 1. 데이터 로드
# =========================
iris = load_iris()
X = iris.data
y = iris.target


# =========================
# 2. 학습 / 테스트 분리
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 클래스 비율 유지 (중요)
)


# =========================
# 3. 모델 정의
# =========================
models = {
    "logistic": LogisticRegression(max_iter=200),
    "tree": DecisionTreeClassifier(random_state=42),
    "forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# =========================
# 4. 모델 학습 + 평가
# =========================
trained_models = {}

print("=== Iris Model Training Start ===")

for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[{name}] accuracy: {acc:.4f}")

    trained_models[name] = model


# =========================
# 5. 모델 저장
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(trained_models["logistic"],
            os.path.join(MODEL_DIR, "iris_logistic.pkl"))

joblib.dump(trained_models["tree"],
            os.path.join(MODEL_DIR, "iris_tree.pkl"))

joblib.dump(trained_models["forest"],
            os.path.join(MODEL_DIR, "iris_forest.pkl"))

print("\n=== Models saved ===")
print(" - iris_logistic.pkl")
print(" - iris_tree.pkl")
print(" - iris_forest.pkl")
