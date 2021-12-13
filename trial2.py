import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential
from tensorflow.python import keras
from tensorflow.python.keras.layers import BatchNormalization, Flatten, Convolution2D, Convolution2DTranspose, Dropout
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.np_utils import to_categorical

width, height = 32, 32
batch_size = 1000
epochs = 10
num_classes = 10
validation_split = 0.25
verbosity = 1

'''Load Dataset'''
(X_all_train, y_all_train), (X_test, y_test) = cifar10.load_data()

'''Encoding'''
y_all_train = to_categorical(y_all_train)
y_test = to_categorical(y_test)

'''Parse to float32'''
X_all_train = X_all_train.astype('float32')
X_test = X_test.astype('float32')

'''Normalization'''
X_all_train, X_test = X_all_train / 255.0, X_test / 255.0

'''Model Creation'''
model = Sequential()

model.add(Conv2D(3, kernel_size=(1, 1), activation='relu', padding='same', input_shape=(32, 32, 3)))

model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))

model.add(Conv2DTranspose(16, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(32, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(64, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))

# model.add(keras.layers.Dense(3, activation='softmax'))
model.add(Conv2D(3, kernel_size=(1, 1), activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

'''Fitting'''
history = model.fit(X_all_train, X_all_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split)

'''Plot Accuracy & Loss'''
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss'], loc='upper left')
plt.show()

pd.DataFrame(history.history).plot()

'''Prediction'''
# model.evaluate(X_test, y_test)
#
# pred = model.predict(X_test)
# y_pred = np.argmax(pred, axis=1)
# acc = np.mean(y_pred == y_test)
# print("accuracy: {}".format(acc))

'''Weights'''
for w in model.trainable_weights:
    print(K.eval(w))

'''Plotting Sample Images'''

num_reconstructions = 8
samples = X_test[:num_reconstructions]
targets = y_test[:num_reconstructions]
reconstructions = model.predict(samples)

for i in np.arange(0, num_reconstructions):
    sample = samples[i][:, :, 0]
    reconstruction = reconstructions[i][:, :, 0]
    input_class = targets[i]
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(sample)
    axes[0].set_title('Original image')
    axes[1].imshow(reconstruction)
    axes[1].set_title('Reconstruction with Conv2DTranspose')
    fig.suptitle(f'CIFAR Target = {input_class}')
    plt.show()
