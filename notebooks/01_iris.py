# %%

# Iris 데이터셋 분류 예제 (1번)
# 목적: 머신러닝 전체 흐름을 한 번에 실습
# - 데이터 로딩
# - 간단한 확인(EDA)
# - 학습/테스트 분리
# - 모델 학습
# - 평가 및 시각화

# ================================
# 1. 라이브러리 임포트
# ================================
# 데이터 처리
import pandas as pd


# 시각화
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터셋 로드
from sklearn.datasets import load_iris

# 데이터 분리 (train / test)
from sklearn.model_selection import train_test_split

# 사용할 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 평가 지표
from sklearn.metrics import accuracy_score, classification_report

# ================================
# 2. 데이터 로딩
# ================================
# scikit-learn 내장 Iris 데이터셋 로드
iris = load_iris()

# 특성(feature)을 DataFrame으로 변환(입력값)
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# 타깃(label)(정답값)
y = iris.target

# 클래스 이름 확인
# iris.target_names -> ['setosa', 'versicolor', 'virginica']
print("Class names:", iris.target_names)


# ================================
# 3. 데이터 기본 확인 (EDA)
# ================================
# 상위 5개 데이터 확인
print(X.head())

# 데이터 크기 확인
print("X shape:", X.shape)
print("y shape:", y.shape)

# 기본 통계량 확인
print(X.describe())

# ================================
# 3-1. EDA 시각화 (pairplot)
# ================================
df = X.copy()
df["species"] = y
df["species"] = df["species"].map(dict(enumerate(iris.target_names)))

sns.pairplot(df, hue="species")
plt.show()


# ================================
# 3-2. 주요 feature 산점도
# ================================
sns.scatterplot(
    x=X["petal length (cm)"],
    y=X["petal width (cm)"],
    hue=y
)
plt.show()


# ================================
# 4. 학습 / 테스트 데이터 분리
# ================================
# test_size=0.2 → 전체 데이터의 20%를 테스트용으로 사용
# random_state=42 → 항상 같은 결과가 나오도록 고정 (재현성)
# stratify=y → 클래스 비율 유지 (분류 문제에서 중요)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ==================================================
# 5. 모델 정의
# ==================================================
# 비교할 모델들을 딕셔너리 형태로 정의
# key : 모델 이름 (출력용)
# value : 실제 모델 객체
models = {
"Logistic Regression": LogisticRegression(max_iter=200),
# 선형 모델, 기준선(baseline)으로 사용

"Decision Tree": DecisionTreeClassifier(random_state=42),
# 비선형 모델, 과적합되기 쉬움

"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
# 여러 트리를 결합한 앙상블 모델, 안정적인 성능
}


# ==================================================
# 5. 모델 학습 및 평가
# ==================================================
# 결과를 저장할 리스트
results = []

# 모든 모델을 동일한 방식으로 반복 실행
for name, model in models.items():
    
    # ------------------------------
    # (1) 모델 학습
    # ------------------------------
    model.fit(X_train, y_train)
    # ------------------------------
    # (2) 예측
    # ------------------------------
    y_pred = model.predict(X_test)
    # ------------------------------
    # (3) 정확도 계산
    # ------------------------------
    acc = accuracy_score(y_test, y_pred)


    # 결과 저장 (나중에 시각화용)
    results.append({
    "model": name,
    "accuracy": acc
    })


    # ------------------------------
    # (4) 결과 출력
    # ------------------------------
    print("=" * 50)
    print(f"Model: {name}")
    print("Accuracy:", acc)


    # precision / recall / f1-score 출력
    print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
    ))
    
    '''
    precision (정밀도)
    - 모델이 특정 클래스로 예측한 것 중 실제로 맞은 비율
    - false positive(오탐)가 얼마나 적은지를 의미

    recall (재현율)
    - 실제 해당 클래스 중 모델이 맞게 찾아낸 비율
    - false negative(놓침)가 얼마나 적은지를 의미

    f1-score (★★★ 중요)
    - precision과 recall의 조화 평균
    - 모델 성능을 한 줄로 요약하는 핵심 지표

    accuracy
    - 전체 데이터 중 모델이 맞춘 비율


    [결과 해석]

    Iris 데이터셋에서
    Logistic Regression 모델은
    모든 클래스에서 precision과 recall이 0.9 이상으로 높게 나타났으며,
    특정 클래스에 치우치지 않고
    전반적으로 안정적인 분류 성능을 보였다.

    '''
    
# ==================================================
# 6. 모델 성능 비교 시각화
# ==================================================
# 결과 리스트를 DataFrame으로 변환
results_df = pd.DataFrame(results)


# 막대그래프로 모델별 정확도 비교
plt.figure(figsize=(6, 4))
sns.barplot(data=results_df, x="model", y="accuracy")


plt.ylim(0, 1)
plt.title("Model Accuracy Comparison (Iris)")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.tight_layout()
plt.show()

# ==================================================
# 7. 해석 요약 (주석)
# ==================================================
'''
[모델별 해석 요약]


1) Logistic Regression
- 선형 모델
- 구조가 단순한 데이터에서는 높은 성능
- 해석이 쉽고 기준선 모델로 적합


2) Decision Tree
- 비선형 패턴 학습 가능
- 학습 데이터에 과적합되기 쉬움
- 파라미터(max_depth 등) 조정이 중요


3) Random Forest
- 여러 Decision Tree의 결과를 평균
- 과적합을 줄이고 일반화 성능이 뛰어남
- 실무에서 기본 선택으로 자주 사용


[Iris 데이터셋 결론]
- 세 모델 모두 높은 성능을 보임
- 데이터가 비교적 단순한 구조
- Random Forest가 가장 안정적인 결과를 제공
'''


# ================================
# 8. 혼동 행렬(Confusion Matrix) 시각화
# ================================
# 어떤 클래스를 헷갈리는지 확인하기 위함
# Confusion Matrix = 모델의 오답 노트
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Iris)")
plt.tight_layout()
plt.show()

'''
[실무 적용 관점에서의 머신러닝 전체 과정 정리]

1. 문제 정의 (Business Understanding)
- 단순히 "정확도 높은 모델"이 목표가 아님
- 비즈니스 관점에서 예측 대상(y)을 명확히 정의
  예)
  - 고객 이탈 여부 예측
  - 사기 거래 탐지
  - 특정 행동 발생 여부 예측

2. 데이터 이해 및 EDA (Exploratory Data Analysis)
- 데이터 분포, 클래스 불균형 여부 확인
- feature 간 관계 시각화(pairplot, scatter 등)
- 데이터만 보고도 분리가 가능한 문제인지 판단
- 이 단계에서 모델의 한계를 예측할 수 있음

3. 데이터 분리 (Train / Test)
- 학습 데이터와 평가 데이터를 명확히 분리
- 데이터 누수(data leakage) 방지
- stratify 옵션으로 클래스 비율 유지
- random_state 고정으로 재현성 확보

4. 모델 선택 및 비교
- 기준선 모델(Logistic Regression)부터 시작
- 비선형 모델(Decision Tree) 적용
- 앙상블 모델(Random Forest)로 성능 안정화
- 동일한 데이터, 동일한 조건에서 공정하게 비교

5. 성능 평가 (Metrics)
- Accuracy만으로 모델을 판단하지 않음
- Precision, Recall, F1-score를 함께 확인
- 문제 특성에 따라 더 중요한 지표를 선택
  예)
  - 사기/의료: Recall 중요
  - 추천/알림: Precision 중요

6. 혼동 행렬(Confusion Matrix) 분석
- 모델이 어떤 클래스를 자주 틀리는지 확인
- False Positive / False Negative의 비즈니스 영향 분석
- 단순 점수보다 실제 리스크 파악이 목적

7. 모델 선택 근거 정리
- 가장 점수가 높은 모델이 아니라
  목적에 가장 적합한 모델을 선택
- 성능 + 안정성 + 해석 가능성 고려

8. 실무 확장
- Pipeline으로 전처리 + 모델 일관성 유지
- 모델 저장(joblib/pickle) 후 재사용
- Flask/FastAPI와 연동하여 예측 API 구성
- 새로운 데이터에도 동일한 파이프라인 적용

[정리]
본 실습은 Iris 데이터셋을 사용했지만,
실무 머신러닝 프로젝트의 전체 흐름을
축소하여 경험하는 것을 목표로 한다.
'''


# %%
