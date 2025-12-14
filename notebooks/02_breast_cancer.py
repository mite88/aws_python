# %%
"""
[2번째 실습]
Breast Cancer Wisconsin (Diagnostic) 데이터셋
- 문제 유형: 이진 분류
- 목표: 종양이 악성(malignant)인지 양성(benign)인지 예측
- 핵심 포인트:
  1) 스케일링 필수
  2) stratify 분할
  3) 의료 데이터 → Recall 중요
  4) Pipeline + 모델 저장
"""

# =========================
# 0. 라이브러리 임포트
# =========================
import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1. 경로 설정 (실무 정석)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 2. 데이터 로딩
# =========================
data = load_breast_cancer()

#X = data.data          # 입력 피처 (30개)
# mean 계열 10개만 사용
X = data.data[:, :10]
y = data.target        # 타겟 (0=악성, 1=양성)

# 데이터프레임으로 확인 (EDA용)
#df = pd.DataFrame(X, columns=data.feature_names)
df = pd.DataFrame(
    X,
    columns=data.feature_names[:10]  # ⭐ mean 컬럼 10개만
)
df["target"] = y

# =========================
# 3. 데이터 기본 확인
# =========================
print("=== 데이터 크기 ===")
print(df.shape)

print("\n=== 타겟 분포 ===")
print(df["target"].value_counts())

# =========================
# 4. 학습 / 테스트 분리
# =========================
# stratify=y → 악성/양성 비율 유지 (의료 데이터 필수)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. 파이프라인 구성
# =========================
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),     # 피처 스케일링
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# =========================
# 6. 모델 학습
# =========================
pipe_lr.fit(X_train, y_train)

# =========================
# 7. 예측
# =========================
y_pred = pipe_lr.predict(X_test)

# =========================
# 8. 평가
# =========================
print("\n=== Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

'''
[[TN FP]
 [FN TP]]
 
[[41  1]
 [ 1 71]]
 
 FN (악성 → 양성 예측)
-> 의료 데이터에서 가장 위험한 오차

Accuracy 높아도 FN 많으면 나쁜 모델
'''

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

'''
| 지표        | 의미                    | 중요도 |
| --------- | --------------------- | --- |
| Precision | 양성이라 한 것 중 진짜         | 보통  |
| Recall    | 진짜 양성 중 맞춘 비율         | ⭐⭐⭐ |
| F1-score  | Precision + Recall 균형 | ⭐⭐  |
| Accuracy  | 전체 정답률                | 참고  |

-> Recall 낮으면 모델 다시 설계해야 함
'''

# 오차율(Error Rate)
error_rate = 1 - pipe_lr.score(X_test, y_test)
print("\nError Rate:", error_rate)

# =========================
# 9. 모델 저장
# =========================
model_path = os.path.join(MODEL_DIR, "cancer_logistic.pkl")
joblib.dump(pipe_lr, model_path)

print(f"\n모델 저장 완료: {model_path}")

# %%
