import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential
from tensorflow.python import keras
from tensorflow.python.keras.layers import BatchNormalization, Flatten, Convolution2D, Convolution2DTranspose, \
    MaxPooling2D, Dropout, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainY = to_categorical(trainY)
testY = to_categorical(testY)
trainX, trainY, testX, testY


train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')

train_norm = train_norm / 255.0
test_norm = test_norm / 255.0

train_norm, test_norm

'''Sample Images'''
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(X_all_train[i])
#     plt.xlabel(class_names[y_all_train[i][0]])
# plt.show()

'''Model Creation'''
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                 input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

'''Fitting'''
history = model.fit(trainX, trainY, epochs=10, verbose=1, validation_data=(testX, testY))
acc = model.evaluate(testX, testY, verbose=0)
print('> %.3f' % (acc * 100.0))

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

'''Weights'''
for w in model.trainable_weights:
    print(K.eval(w))

'''Plotting Sample Images'''

# num_reconstructions = 8
# samples = X_test[:num_reconstructions]
# targets = y_test[:num_reconstructions]
# reconstructions = model.predict(samples)
#
# for i in np.arange(0, num_reconstructions):
#     sample = samples[i][:, :, 0]
#     reconstruction = reconstructions[i][:, :, 0]
#     input_class = targets[i]
#     fig, axes = plt.subplots(1, 2)
#     axes[0].imshow(sample)
#     axes[0].set_title('Original image')
#     axes[1].imshow(reconstruction)
#     axes[1].set_title('Reconstruction with Conv2DTranspose')
#     fig.suptitle(f'CIFAR Target = {input_class}')
#     plt.show()
