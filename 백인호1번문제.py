# -*- coding: utf-8 -*-

# === 1. 라이브러리 임포트 ===
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import platform

# === 2. 데이터 불러오기 및 전처리 ===
# 파일 경로가 상대 경로이므로 스크립트 실행 위치를 확인해야 합니다.
try:
    df = pd.read_csv('1번문제/Multi-class Classification Datasets.csv', sep=';')
except FileNotFoundError:
    print("오류: '1번문제/Multi-class Classification Datasets.csv' 파일을 찾을 수 없습니다.")
    print("스크립트 파일과 데이터 파일의 위치를 확인해주세요.")
    exit() # 파일이 없으면 실행 중단

# 데이터 확인
print("데이터 전처리 전 데이터 정보")
print(df.info())

# 컬럼명 공백 및 특수문자 제거
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('"', '')
df.columns = df.columns.str.replace('\t', '')
print("\n공백 및 특수문제 제거 후 데이터 정보")
print(df.info())

print("\n--- 데이터 정보 ---")
print("타겟 변수 분포:")
print(df['Target'].value_counts())
print("\n")

# === 3. 데이터 시각화 (한글 폰트 설정 포함) ===
# 시스템에 따라 폰트 지정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'  # 리눅스나 구글 코랩 등

# 마이너스(-) 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

sns.countplot(x='Target', data=df)
plt.title("Target 클래스 분포")
plt.show()

# === 4. 범주형 데이터 인코딩 ===
# dtype이 'object'인 열만 확인 (문자형 열)
cat_cols = df.select_dtypes(include='object').columns
print("범주형 컬럼:", cat_cols.tolist())

# 각 문자형 열에 대해 LabelEncoder 적용
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n--- 데이터 타입 확인 ---")
print(df.dtypes)
print("\n")

# === 5. 피처(X)와 타겟(y) 분리 ===
X = df.drop('Target', axis=1)
y = df['Target']

# === 6. 훈련/테스트 데이터 분할 ===
# stratify=y를 설정하면 타깃 비율을 train/test에 동일하게 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=100
)

print("--- 데이터 분할 결과 ---")
print("Train 데이터:", X_train.shape, y_train.shape)
print("Test 데이터:", X_test.shape, y_test.shape)
print("Train 클래스 분포:\n", y_train.value_counts(normalize=True))
print("Test 클래스 분포:\n", y_test.value_counts(normalize=True))
print("\n")

# === 7. 데이터 스케일링 (ConvergenceWarning 해결 및 성능 향상) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일링 후 DataFrame으로 다시 변환 (컬럼명 유지를 위해)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# === 8. 모델 및 하이퍼파라미터 그리드 정의 ===
# 1. 랜덤 포레스트
rfc = RandomForestClassifier(random_state=100)
param_rfc = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None]
}

# 2. XGBoost
xgb = XGBClassifier(eval_metric='mlogloss', random_state=100, use_label_encoder=False)
param_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.01]
}

# 3. 로지스틱 회귀
lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=100)
param_lr = {
    'C': [0.01, 1, 10]
}

# === 9. GridSearchCV를 사용한 모델 학습 및 최적화 ===
print("--- 모델 학습 및 하이퍼파라미터 최적화 시작 ---")
# 랜덤 포레스트 그리드 서치
grid_rfc = GridSearchCV(rfc, param_rfc, cv=3, scoring='f1_macro', n_jobs=-1)
grid_rfc.fit(X_train_scaled, y_train)
print("Random Forest 최적화 완료")

# XGBoost 그리드 서치
grid_xgb = GridSearchCV(xgb, param_xgb, cv=3, scoring='f1_macro', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)
print("XGBoost 최적화 완료")

# 로지스틱 회귀 그리드 서치
grid_lr = GridSearchCV(lr, param_lr, cv=3, scoring='f1_macro', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
print("Logistic Regression 최적화 완료")
print("\n")

# === 10. 최적 파라미터 및 성능 평가 ===
print("--- 최적 하이퍼파라미터 ---")
print("랜덤포레스트 최적 파라미터:", grid_rfc.best_params_)
print("XGBoost 최적 파라미터:", grid_xgb.best_params_)
print("로지스틱회귀 최적 파라미터:", grid_lr.best_params_)
print("\n")

models = {
    "Random Forest": grid_rfc.best_estimator_,
    "XGBoost": grid_xgb.best_estimator_,
    "Logistic Regression": grid_lr.best_estimator_
}

print("--- 모델별 분류 리포트 ---")
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} 분류 리포트:\n")
    print(classification_report(y_test, y_pred))

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"▶ {name} - Accuracy: {acc:.4f}, Macro F1-score: {f1:.4f}")
    return acc, f1

print("\n--- 모델별 최종 성능 요약 ---")
results = {}
for name, model in models.items():
    acc, f1 = evaluate_model(model, X_test_scaled, y_test, name)
    results[name] = {'Accuracy': acc, 'F1_score': f1}

# 타깃 데이터 바이너리 형태로 변환 (ROC AUC 계산용)
y_test_bin = label_binarize(y_test, classes=np.unique(y))

print("\n--- 모델별 ROC AUC 점수 ---")
for name, model in models.items():
    y_prob = model.predict_proba(X_test_scaled)
    auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')
    print(f"▶ {name} - ROC AUC Score (OvR): {auc:.4f}")

# === 11. 결과 시각화 ===
# 모델 성능 비교 시각화
df_result = pd.DataFrame(results).T
df_result.plot(kind='bar', figsize=(8, 5))
plt.title('모델별 Accuracy 및 F1-score 비교')
plt.xticks(rotation=0)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 혼동 행렬 시각화
def plot_confusion(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

for name, model in models.items():
    plot_confusion(model, X_test_scaled, y_test, name)

# XGBoost 모델에서 중요도 추출
xgb_model = models["XGBoost"]
importances = xgb_model.feature_importances_

# 스케일링 전의 X 컬럼명을 사용
feature_names = X.columns

# 중요도 순 정렬
sorted_idx = importances.argsort()[::-1]
top_n = 15  # 상위 15개 특성만 시각화

plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances[sorted_idx][:top_n],
    y=feature_names[sorted_idx][:top_n]
)
plt.title("XGBoost - 주요 Feature Importance (Top 15)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 더 체계적이고 재사용 가능한 코드를 작성하기 위해 main() 함수를 사용하는게 좋음