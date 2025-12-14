# ML Study (Flask 기반 머신러닝 실습)

## 1. 기본 설명

- 목적  
  - 머신러닝 실습 → 모델 학습 → Flask API → HTML 서비스까지  
    **실무 흐름을 하나의 프로젝트 안에서 반복 학습**
- 실행 환경  
  - OS: Ubuntu  
  - Python 가상환경: `flask`  
  - 실행 방식: CLI + Flask Web

---

## 2. 개발 환경

### 가상환경 생성 및 활성화

```bash
python3 -m venv flask
source flask/bin/activate
```

### Python 버전

```bash
python --version
# Python 3.12.x
```

---

## 3. 라이브러리 설치 순서 (중요 ⭐)

### 1️⃣ pip 업그레이드

```bash
pip install --upgrade pip
```

### 2️⃣ 필수 라이브러리 설치

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter flask joblib
```

### 3️⃣ Ubuntu 시각화 관련 패키지

```bash
sudo apt install -y python3-tk
```

> 📌 주의  
> - matplotlib 시각화는 **노트북(실습 단계)** 에서만 사용  
> - Flask 서비스에서는 시각화가 아닌 **예측 결과만 제공**

---

## 4. 프로젝트 파일 구조

```text
ml-study/
├── app.py                # Flask 서버 시작점 (절대 삭제 X)
│
├── ml/                   # 머신러닝 학습 코드
│   ├── iris_train.py     # Iris 모델 학습
│   ├── iris_utils.py     # 클래스명, 공통 유틸
│   ├── cancer_train.py   # Breast Cancer 모델 학습
│   └── housing_train.py  # Housing 회귀 모델 학습
│
├── models/               # 학습된 모델 저장소
│   └── iris_model.pkl
│
├── notebooks/            # 실습 / 분석용 코드
│   ├── 01_iris.py
│   ├── 02_breast_cancer.py
│   ├── 03_wine.py
│   ├── 04_housing.py
│   └── 05_digits.py
│
├── routes/               # Flask 라우트 (모델별 분리)
│   ├── iris.py
│   ├── cancer.py
│   └── housing.py
│
├── templates/            # HTML 템플릿
│   ├── base.html
│   ├── index.html
│   ├── iris.html         ← Iris 입력/결과
│   ├── cancer.html
│   └── housing.html
│
├── static/               # 정적 리소스
│   ├── css/style.css
│   └── js/
│
└── README.md
```

---

## 5. 실행 순서 (중요 ⭐)

### 1️⃣ 가상환경 활성화

```bash
source flask/bin/activate
```

---

### 2️⃣ 실습 코드 확인 / 실행 (분석 단계)

```bash
# Jupyter 실행
jupyter notebook
```

- notebooks/01_iris.py
- notebooks/02_breast_cancer.py
- notebooks/03_wine.py
....

👉 데이터 분포, 시각화, 모델 비교 목적

---

### 3️⃣ 모델 학습 (train 스크립트 실행)

```bash
python ml/iris_train.py
```

정상 실행 시:
- models/iris_logistic.pkl
- models/iris_tree.pkl
- models/iris_forest.pkl

파일이 생성됨

---

### 4️⃣ Flask 서버 실행 (서비스 단계)

```bash
python app.py
```


* 서버에서 자동 실행하기 위한 방법

```
pip install gunicorn

sudo nano /etc/systemd/system/flask.service
```

```
[Unit]
Description=Flask ML Study Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/flask/ml-study
Environment="PATH=/home/ubuntu/flask/bin"
ExecStart=/home/ubuntu/flask/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```
sudo systemctl daemon-reload
sudo systemctl start flask
sudo systemctl enable flask

# 상태확인
sudo systemctl status flask

# 재실행
sudo systemctl restart flask.service

```
---

### 5️⃣ 브라우저 접속

```text
http://127.0.0.1:5000/
or
http://13.237.30.109:5000/
```

- HTML 입력 폼 표시
- 예측 결과 + 모델별 확률 확인 가능

---

## 6. 실무 기준 학습 흐름

```text
[노트북 / .py 실습]
   ↓
[EDA + 모델 비교]
   ↓
[train_*.py로 최종 모델 학습]
   ↓
[pkl 파일 저장]
   ↓
[Flask에서 모델 로드]
   ↓
[HTML / API 제공]
```

> ✔ 실습 코드와 서비스 코드를 **명확히 분리**하는 것이 핵심

---

## 6. 파일 종류별 역할 정리

| 파일 종류 | 목적 | 실행 / 확인 방법 |
|---------|------|----------------|
| `*.ipynb. .py` | 데이터 이해, 시각화, 흐름복습 | Jupyter, VSCode |
| `*_train.py` | 모델 학습 | 터미널 |
| `app.py` | Flask 서버 실행 | 터미널 |
| `routes/*.py` | 웹 요청 처리 | Flask |
| `index.html` | 사용자 화면 | 브라우저 |
| `*.pkl` | 학습된 모델 | 코드로 로드 |

---

## 7. 두 번째 / 세 번째 실습 확장 규칙 (중요 ⭐)

### ✔ 새로운 데이터셋 추가 시 반드시 할 것

#### 1️⃣ notebooks

```text
notebooks/
└── 06_new_dataset.py
```

- EDA
- 시각화
- 모델 비교
- 해석 주석 작성

---

#### 2️⃣ ml

```text
ml/
└── new_dataset_train.py
```

- 최종 모델 학습
- `models/new_dataset_model.pkl` 생성

---

#### 3️⃣ routes

```text
routes/
└── new_dataset.py
```

- 입력 폼 처리
- 예측 API / HTML 연결

---

#### 4️⃣ templates

```text
templates/
└── new_dataset.html
```

> 👉 위 구조만 지키면 **데이터셋 개수와 상관없이 확장 가능**

---

## 8. 실무 관점 체크리스트

- [ ] 가상환경에서 학습과 서비스 실행을 모두 수행했는가?
- [ ] 학습 환경과 서비스 환경이 동일한가?
- [ ] 모델(pkl)과 코드가 분리되어 있는가?
- [ ] 입력값 검증 로직을 추가할 수 있는 구조인가?
- [ ] README에 실습 목적과 결과를 정리했는가?

---

## 9. 앞으로 추가 예정 기능

- [ ] 예측 확률(probability) 반환
- [ ] 입력값 validation
- [ ] 시각화 결과 이미지 저장
- [ ] 여러 모델 선택 UI
- [ ] FastAPI 버전 구현
- [ ] Docker 기반 배포

---

## 10. 프로젝트 한 줄 요약

> 이 프로젝트는  
> **머신러닝 실습을 서비스 관점으로 확장하기 위한 학습용 베이스 프로젝트**이다.

