# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"

heart_df = pd.read_csv(file_url)

print( 'Pandas DataFrame :', type(heart_df) )

# pandas dataFrame
print( heart_df.shape )

heart_df.head()

import tensorflow as tf
from tensorflow.data.experimental import make_csv_dataset

file_name = "/content/drive/MyDrive/Colab Notebooks/data/Cleveland Clinic Foundations for Heart Disease/heart.csv"
heart_ds = make_csv_dataset(file_name, batch_size=2)

print( 'tf.data : ', type(heart_ds) )

iterator = heart_ds.as_numpy_iterator()

print( dict(next(iterator)) )

df = heart_df.copy()
label = df.pop('target')

ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
ds = ds.batch(2)

print( 'tf.data : ', type(ds) )

print(list(ds.as_numpy_iterator()))

import tensorflow as tf
from tensorflow.keras.utils import get_file
import os
#import pathlib

train_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

def get_data(fname, origin, extract):
    data_dir = get_file(fname=fname, origin=origin, extract=extract)
    data_dir = os.path.join(os.path.dirname(data_dir), 'cats_and_dogs_filtered')
    return data_dir

path_dir = get_data('train.zip', train_url, 'True')

train_dir = os.path.join(path_dir, 'train')
validation_dir = os.path.join(path_dir, 'validation')

print(path_dir, '\n')
print(train_dir, '\n')
print(validation_dir)

# the Cats vs Dogs dataset
!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

!unzip -q kagglecatsanddogs_3367a.zip

from tensorflow.keras.preprocessing import image_dataset_from_directory

PetImages_ds = image_dataset_from_directory(
  directory=train_dir,
  labels='inferred',
  batch_size=32,# Default: 32
  image_size=(256, 256),# Defaults: (256, 256)
  shuffle=True,# Default: True
  seed=0)

print("Label 0 : ", PetImages_ds.class_names[0])
print("Label 1 : ", PetImages_ds.class_names[1])

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for image, label in PetImages_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(image[i].numpy().astype("uint8"))
        plt.title(int(label[i]))
        plt.axis("off")

# Load the data: IMDB movie review sentiment classification
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

%%time
!tar -xf aclImdb_v1.tar.gz

!du -h /content/aclImdb

from tensorflow.keras.preprocessing import text_dataset_from_directory

aclImdb_ds = text_dataset_from_directory(
    directory = "/content/aclImdb/train",
    labels="inferred",
    batch_size=32,
    seed=0
)

print("Label 0 : ", aclImdb_ds.class_names[0])
print("Label 1 : ", aclImdb_ds.class_names[1])
print("Label 2 : ", aclImdb_ds.class_names[2])

for text, label in aclImdb_ds.take(1):
    for i in range(2):
        print(text.numpy()[i])
        print(label.numpy()[i])


