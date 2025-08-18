# Pizza or Not Pizza 데이터 셋 대상으로 합성곱신경망을 사용하여 이미지 분류예측을 실행
# 하고 parameter 최적화를 통해 최적의 분석모델 수립하시오. 분류성능지표(accuracy, f1
# score, ROC-AUC 등)를 산출하고 insight를 도출할 수 있도록 시각화 하시오.
# ◼ Pizza or Not Pizza datatset
# ===================
# import kagglehub
# # Download latest version
# path = kagglehub.dataset_download("carlosrunner/pizza-not-pizza")
# print("Path to dataset files:", path)
# ===========================
# ◼ Training dataset과 Test dataset의 규모는 팀별로 자율적으로 결정하시오.
# ◼ Train/Test dataset 분할비율은 default 사용(80%-20%)
# ◼ sklearn.model_selection.train_test_split( ) 함수내 random_state는  100으로 설정