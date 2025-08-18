# 고등교육기관에서 생성한 데이터셋을 분석하여 정규과정이 끝나는 시점에서의 결과
# (dropout, enrolled, and graduate)의 다중분류예측을 위하여 3가지 이상 다중분류모델을
# 사용하고 parameter 최적화를 통해 최적의 분석모델 수립하시오. 분류성능지표(accuracy,
# f1 score, ROC-AUC 등)를 산출하고 insight를 도출할 수 있도록 시각화 하시오.
# ◼ Dataset:
# https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+suc
#  cess
# ◼ Train/Test dataset 분할비율은 default 사용(80%-20%)
# ◼ Class들의 균등분포를 위하여 stratify 옵션 사용
# ◼ sklearn.model_selection.train_test_split( ) 함수내 random_state는  100으로 설정

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 가져오기
df = pd.read_csv('data.csv', sep=';')

# 칼럼 및 타입 확인
df.info()
# 결측치 확인
df.isnull().sum()

# 종속변수(Target) 값별 개수 확인
df.Target.value_counts()

# 종속변수가 범주형 > 숫자형 데이터로 변환 (라벨 인코딩)
label_encoder = LabelEncoder()
df['Target_encoded'] = label_encoder.fit_transform(df['Target'])
class_names = label_encoder.classes_  # ['Dropout', 'Enrolled', 'Graduate']
# 종속변수 숫자형 전환 확인 (0 = Dropout, 1 = Enrolled, 2 = Graduate)
df['Target_encoded'].value_counts()

# Feature / Target 분리
X = df.drop(['Target','Target_encoded'], axis = 1)
y = df['Target_encoded']

# 데이터 분할
# Class들의 균등분포를 위하여 stratify 옵션 사용
# sklearn.model_selection.train_test_split( ) 함수내 random_state는  100으로 설정
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=100 # stratify = y : 클래스 비율 유지
)

# 데이터 스케일링
scaler = StandardScaler() # 표준화 객체
X_train_scaled = scaler.fit_transform(X_train) # 학습 데이터를 통계정보에 맞춰 변환
X_test_scaled = scaler.transform(X_test) # 평가용은 모델 평가를 위해 fit 사용 X

# 모델과 하이퍼파라미터 튜닝
model = {'RandomForest' : RandomForestClassifier(),
         'XGBoost' : XGBClassifier()
         }
param = {'RandomForest' : {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [None, 10, 20],
    'min_samples_split': [2, 5]
    },
    'XGBoost' : {}
}

# 모델 꺼내기
rf_model = model['RandomForest']

# 파라미터 그리드 꺼내기
param_grid = param['RandomForest']

# GridSearchCV 객체 생성
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='f1_weighted',   # f1 weighted 사용
    cv=5,
    n_jobs=-1,
)

# 학습 데이터로 그리드 서치 실행
grid_search.fit(X_train_scaled, y_train)

# 최적의 모델 저장
best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# 테스트 데이터 예측 및 예측 확률
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)

# 평가 지표 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 다중분류용 원-핫 인코딩 (ROC-AUC 계산용)
classes = best_model.classes_
y_test_bin = label_binarize(y_test, classes=classes)

# ROC-AUC (One-vs-Rest) 계산 및 출력
roc_auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
print("ROC AUC (weighted, OvR):", roc_auc)

# 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 클래스별 ROC curve 반복문 시각화
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc='lower right')
plt.grid()
plt.show()