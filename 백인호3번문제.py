# -*- coding: utf-8 -*-

# === 1. 라이브러리 불러오기 ===
import os
import glob
import shutil
import kagglehub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def main():
    # === 2. 데이터 다운로드 ===
    print("--- 1. Kaggle 데이터셋 다운로드 시작 ---")
    try:
        # KaggleHub를 통해 데이터셋 다운로드 (인증 필요)
        path = kagglehub.dataset_download("carlosrunner/pizza-not-pizza")
        print("데이터 다운로드 경로:", path)

        # 실제 이미지 폴더 경로 확인
        pizza_dir = os.path.join(path, "pizza_not_pizza", "pizza")
        not_pizza_dir = os.path.join(path, "pizza_not_pizza", "not_pizza")

        if not os.path.exists(pizza_dir) or not os.path.exists(not_pizza_dir):
            raise FileNotFoundError("다운로드된 데이터셋에서 'pizza_not_pizza' 폴더를 찾을 수 없습니다.")

    except Exception as e:
        print(f"오류: 데이터 다운로드에 실패했습니다. {e}")
        print("Kaggle API 인증(kaggle.json)이 올바르게 설정되었는지 확인해주세요.")
        return
    print("--- 데이터셋 다운로드 완료 ---\n")

    # === 3. Train/Test 폴더 생성 및 데이터 분할 ===
    print("--- 2. Train/Test 데이터 분할 시작 ---")
    # 이미지 경로 리스트 생성
    pizza_images = glob.glob(os.path.join(pizza_dir, "*.jpg"))
    not_pizza_images = glob.glob(os.path.join(not_pizza_dir, "*.jpg"))

    # 클래스별로 Train/Test 분할 (80:20)
    pizza_train, pizza_test = train_test_split(pizza_images, test_size=0.2, random_state=100)
    not_pizza_train, not_pizza_test = train_test_split(not_pizza_images, test_size=0.2, random_state=100)

    # 데이터를 저장할 새로운 폴더 구조 생성
    base_dir = os.path.join(path, "split_data")
    train_pizza_dir = os.path.join(base_dir, "train/pizza")
    train_not_pizza_dir = os.path.join(base_dir, "train/not_pizza")
    test_pizza_dir = os.path.join(base_dir, "test/pizza")
    test_not_pizza_dir = os.path.join(base_dir, "test/not_pizza")

    for folder in [train_pizza_dir, train_not_pizza_dir, test_pizza_dir, test_not_pizza_dir]:
        os.makedirs(folder, exist_ok=True)

    # 분할된 이미지를 새 폴더로 복사
    def copy_images(file_list, dest_dir):
        for img_path in file_list:
            shutil.copy(img_path, dest_dir)

    copy_images(pizza_train, train_pizza_dir)
    copy_images(pizza_test, test_pizza_dir)
    copy_images(not_pizza_train, train_not_pizza_dir)
    copy_images(not_pizza_test, test_not_pizza_dir)
    print(f"데이터가 '{base_dir}' 폴더에 분할 저장되었습니다.")
    print("--- Train/Test 데이터 분할 완료 ---\n")

    # === 4. ImageDataGenerator로 이미지 불러오기 ===
    print("--- 3. ImageDataGenerator 설정 시작 ---")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # 픽셀 값을 0~1 사이로 정규화
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'  # 이진 분류
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        shuffle=False  # 평가 시에는 순서가 중요하므로 shuffle=False 설정
    )
    print("--- ImageDataGenerator 설정 완료 ---\n")

    # === 5. CNN 모델 정의 ===
    print("--- 4. CNN 모델 정의 ---")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # 과적합 방지를 위한 Dropout
        Dense(1, activation='sigmoid')  # 이진 분류를 위한 sigmoid 활성화 함수
    ])

    model.summary()
    print("\n")

    # === 6. 모델 컴파일 ===
    print("--- 5. 모델 컴파일 ---")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("--- 모델 컴파일 완료 ---\n")

    # === 7. 모델 학습 실행 ===
    print("--- 6. 모델 학습 시작 ---")
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )
    print("--- 모델 학습 완료 ---\n")

    # === 8. 모델 성능 평가 ===
    print("--- 7. 성능 평가 시작 ---")
    # 예측 확률값 계산
    y_pred_prob = model.predict(test_generator)
    # 확률값을 0.5 기준으로 0 또는 1 클래스로 변환
    y_pred = (y_pred_prob > 0.5).astype(int)
    # 실제 정답값
    y_true = test_generator.classes

    # 성능 지표 계산
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['Not Pizza', 'Pizza']))
    print("--- 성능 평가 완료 ---\n")

    # === 9. 결과 시각화 ===
    print("--- 8. 결과 시각화 ---")

    # 학습 곡선 (Accuracy & Loss)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.suptitle("Learning Curves")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 혼동 행렬 (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Pizza', 'Pizza'],
                yticklabels=['Not Pizza', 'Pizza'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


if __name__ == "__main__":
    main()