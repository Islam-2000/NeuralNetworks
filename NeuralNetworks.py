'''Imports'''

import ssl

import matplotlib.pyplot as plt
from keras.datasets import cifar10
from tensorflow.keras import layers, models

ssl._create_default_https_context = ssl._create_unverified_context

'''Data Importing'''
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

'''Encoding'''
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

'''Preparation'''
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

'''Normalization'''
X_train, X_test = X_train / 255.0, X_test / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Sample Images
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(X_train[i])
#     plt.xlabel(class_names[y_train[i][0]])
# plt.show()

model = models.Sequential()

model.add(layers.Dense(100, activation='sigmoid'))
model.add(layers.Dense(100, activation='sigmoid'))
model.add(layers.Dense(100, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(test_acc)
