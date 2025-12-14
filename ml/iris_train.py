# 모델 학습 스크립트

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os


# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target


# 2. 학습 / 테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# 4. 모델 저장
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'iris_model.pkl')
joblib.dump(model, MODEL_PATH)


print('Iris model saved:', MODEL_PATH)