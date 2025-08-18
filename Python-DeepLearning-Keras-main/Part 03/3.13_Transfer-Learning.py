# from google.colab import drive
# drive.mount('/content/drive')

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input

input_tensor = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=input_tensor, 
              weights='imagenet', 
              include_top=False)

model.summary()

model = VGG16(input_tensor=input_tensor, 
              weights='imagenet', 
              include_top=True)

model.summary()

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input

input_tensor = Input(shape=(224, 224, 3))

model = InceptionV3(input_tensor=input_tensor, 
                    weights='imagenet', 
                    include_top=True)

model.summary()


