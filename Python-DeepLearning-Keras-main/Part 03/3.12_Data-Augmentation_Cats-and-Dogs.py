# from google.colab import drive
# drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.utils import get_file
import os

# train_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
#
# def get_data(fname, origin, extract):
#     data_dir = get_file(fname=fname,
#                         origin=origin,
#                         extract=extract,
#                         cache_subdir='/content/drive/MyDrive/Colab Notebooks/data')
#     data_dir = os.path.join(os.path.dirname(data_dir), 'cats_and_dogs_filtered')
#     return data_dir
#
# path_dir = get_data('train.zip', train_url, 'True')
#
# train_dir = os.path.join(path_dir, 'train')
# validation_dir = os.path.join(path_dir, 'validation')
#
# print('\n', train_dir)
# print(validation_dir)

from tensorflow.keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(
  directory="cats_and_dogs_filtered\\train",
  batch_size=32,# Default: 32
  image_size=(256, 256),# Defaults: (256, 256)
  shuffle=True,# Default: True
  seed=0)

validation_ds = image_dataset_from_directory(
  directory="cats_and_dogs_filtered\\validation",
  batch_size=32,# Default: 32
  image_size=(256, 256),# Defaults: (256, 256)
  shuffle=True,# Default: True
  seed=0)

for data, labels in train_ds.take(1):
    print(data.shape)
    print(labels.shape) 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation

data_augmentation = Sequential(
  [
      RandomFlip("horizontal_and_vertical", input_shape=(256, 256, 3)),
      RandomRotation(0.3),
  ]
)

train_ds = train_ds.prefetch(buffer_size=32)
validation_ds = validation_ds.prefetch(buffer_size=32)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense

model = Sequential([
  data_augmentation,
  Conv2D(16, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  
  Flatten(),
  Dense(128, activation='relu'),
  Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=["accuracy"]
)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

callback_EarlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
callback_ModelCheckpoint = ModelCheckpoint(filepath='./ModelCheckpoint.weights.h5',
                           monitor='val_loss',
                           save_weights_only=True,
                           save_best_only=True,
                           verbose=1)

# %%time

history = model.fit(train_ds,
  validation_data=validation_ds,
  epochs=10,
  callbacks=[callback_EarlyStopping, callback_ModelCheckpoint]
)

#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# img_path = '/content/drive/MyDrive/Colab Notebooks/data/cats_and_dogs_filtered/validation/cats/cat.2008.jpg'
img_path = 'cats_and_dogs_filtered/validation/cats/cat.2008.jpg'

img = load_img(img_path, target_size=(256, 256))
img_array = tf.expand_dims(img_to_array(img), 0)

predictions = model.predict(img_array)

print('cat', 100 * (1 - predictions[0]))
print('dog', 100 * predictions[0])


import matplotlib

matplotlib.use('TkAgg',force=True)

import matplotlib.pyplot as plt
img_array /= 255.

plt.figure(figsize=(5, 5))
plt.imshow(img_array[0])
plt.show()

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(specs=[[{"secondary_y": True}]])

# model history
epoch = history.epoch

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Scatter
fig.add_trace(go.Scatter(x=epoch, y=loss, name="loss"),secondary_y=False,)
fig.add_trace(go.Scatter(x=epoch, y=val_loss, name="val_loss"),secondary_y=False,)
fig.add_trace(go.Scatter(x=epoch, y=accuracy, name="accuracy"),secondary_y=True,)
fig.add_trace(go.Scatter(x=epoch, y=val_accuracy, name="val_accuracy"),secondary_y=True,)

# Templates configuration, Default template: 'plotly'
# Available templates: ['ggplot2', 'seaborn', 'simple_white', 'plotly','plotly_white', 'plotly_dark', 'presentation', 'xgridoff','ygridoff', 'gridon', 'none']
fig.update_layout(title_text="<b>Loss/Accuracy of Model</b>", template='plotly')

fig.update_xaxes(title_text="Epoch")
fig.update_yaxes(title_text="Loss", secondary_y=False)
fig.update_yaxes(title_text="Accuracy", secondary_y=True)

fig.show()



