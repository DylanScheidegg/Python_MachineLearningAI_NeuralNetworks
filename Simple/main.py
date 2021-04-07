import tensorflow as tf
from tensorflow import keras # API to write less code
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# Trains on 90% and test on last 10%
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# simplifying image binary values
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    # numbers = neurons
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax') # probability of what it thinks it is
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_images, train_labels, epochs=5) # epochs how many times to view given data

"""test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc: {}".format(test_acc))"""

prediction = model.predict(test_images)

# Show the image
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: {}'.format(class_names[test_labels[i]]))
    plt.title('Prediction: {}'.format(class_names[np.argmax(prediction[i])]))
    plt.show()

# print(class_names[np.argmax(prediction[0])]) # biggest number is the predicted class


"""print(train_images[7])
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()"""