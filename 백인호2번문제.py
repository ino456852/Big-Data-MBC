# -*- coding: utf-8 -*-

# === 1. 라이브러리 임포트 ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === 2. 데이터 불러오기 및 기본 정보 확인 ===
df = pd.read_csv("1번문제/Multi-class Classification Datasets.csv")

print("--- 원본 데이터 정보 ---")
print(df.info())
print("\n")

# === 3. 피처 엔지니어링 및 전처리 ===
# 요구사항에 따라 Date 관련 변수를 모두 제거합니다.
df = df.drop('Date', axis=1)

# 문자형 변수 레이블 인코딩
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("문자형 변수 목록:", cat_cols)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n--- 전처리 후 데이터 샘플 및 타입 ---")
print(df.head())
print(df.info())
print("\n")

# === 4. 데이터 분할 ===
# 예측 대상은 'Demand', 나머지는 입력값
X = df.drop('Demand', axis=1)
y = df['Demand']

# 훈련/테스트 데이터셋 분할 (80%:20%, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=100
)

print("--- 데이터 분할 결과 ---")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)
print("\n")

# === 5. 데이터 스케일링 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DataFrame으로 변환 (컬럼명 유지)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# === 6. 모델 및 하이퍼파라미터 그리드 정의 ===
# 1. Linear Regression
lr = LinearRegression()

# 2. Ridge Regression
ridge = Ridge()
ridge_params = {'alpha': [0.1, 1.0, 10.0]}

# 3. Lasso Regression
lasso = Lasso()
lasso_params = {'alpha': [0.01, 0.1, 1.0]}

# 4. Random Forest
rf = RandomForestRegressor(random_state=100)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

# 5. Gradient Boosting
gb = GradientBoostingRegressor(random_state=100)
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# === 7. 모델 학습 및 최적화 ===
print("--- 모델 학습 및 최적화 시작 ---")

# Linear Regression
lr.fit(X_train_scaled, y_train)
print("Linear Regression 학습 완료")

# Ridge Regression
grid_ridge = GridSearchCV(ridge, ridge_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_ridge.fit(X_train_scaled, y_train)
print("Ridge Regression 최적화 완료")

# Lasso Regression
grid_lasso = GridSearchCV(lasso, lasso_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_lasso.fit(X_train_scaled, y_train)
print("Lasso Regression 최적화 완료")

# Random Forest Regression
grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
print("Random Forest Regression 최적화 완료")

# Gradient Boosting Regression
grid_gb = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_gb.fit(X_train_scaled, y_train)
print("Gradient Boosting Regression 최적화 완료\n")


# === 8. 모델 성능 평가 ===
# MAPE 함수 정의
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

# 평가 지표 계산 함수
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mae, mse, rmse, mape

# 각 모델 예측
y_pred_lr = lr.predict(X_test_scaled)
y_pred_ridge = grid_ridge.best_estimator_.predict(X_test_scaled)
y_pred_lasso = grid_lasso.best_estimator_.predict(X_test_scaled)
y_pred_rf = grid_rf.best_estimator_.predict(X_test_scaled)
y_pred_gb = grid_gb.best_estimator_.predict(X_test_scaled)

# 평가 결과 저장
results = {
    "LinearRegression": evaluate_model(y_test, y_pred_lr),
    "RidgeRegression": evaluate_model(y_test, y_pred_ridge),
    "LassoRegression": evaluate_model(y_test, y_pred_lasso),
    "RandomForest": evaluate_model(y_test, y_pred_rf),
    "GradientBoosting": evaluate_model(y_test, y_pred_gb)
}

# 결과 DataFrame으로 출력
print("--- 모델별 성능 평가 결과 (Date 변수 제외) ---")
metrics_df = pd.DataFrame(results, index=['MAE', 'MSE', 'RMSE', 'MAPE']).T
print(metrics_df)
print("\n")

# === 9. 결과 시각화 ===
# 지표별 비교 시각화
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg', force=True)

# 한글 폰트 설정을 위한 라이브러리 임포트
import platform

# 운영체제에 맞는 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    # 리눅스나 기타 OS의 경우, NanumGothic 등 설치된 한글 폰트 지정
    plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 부호가 깨지는 것을 방지
plt.rcParams['axes.unicode_minus'] = False

# --- MSE를 제외한 지표 비교 시각화 ---
metrics_df_no_mse = metrics_df.drop('MSE', axis=1)
metrics_df_no_mse.plot(kind='bar', figsize=(12, 7))
plt.title("Performance Comparison (without MSE)")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

# --- MSE 지표만 비교 시각화 ---
metrics_df_mse = metrics_df['MSE']
metrics_df_mse.plot(kind='bar', figsize=(10, 6), color='red')
plt.title("MSE Performance Comparison")
plt.ylabel("Mean Squared Error (MSE)")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- 모델별 실제값 vs 예측값 분산도 시각화 ---
# 최적 모델(Random Forest)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('현재 수요')
plt.ylabel('예상 수요')
plt.title('Random Forest: 실제 수요와 예측 수요(날짜 제외)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Gradient Boosting
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Gradient Boosting: Actual vs Predicted Demand (without Date)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Lasso Regression
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lasso, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Lasso Regression: Actual vs Predicted Demand (without Date)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Linear Regression
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='cyan')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Linear Regression: Actual vs Predicted Demand (without Date)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Ridge Regression
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.5, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Ridge Regression: Actual vs Predicted Demand (without Date)')
plt.grid(True)
plt.tight_layout()
plt.show()