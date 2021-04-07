import tensorflow as tf
from tensorflow import keras
import numpy as np
import os.path
from os import path
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Get data
data = keras.datasets.imdb

# Separate data
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# print(train_data[0])

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# TRIM DATA 250 words
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])

# print(decode_review(test_data[0]))
# print(len(test_data[0]), len(test_data[1]))

model = None
if path.exists('model.h5') == False:
    # MODEL
    best = 0
    for x in range(3):
        model = keras.Sequential([
            keras.layers.Embedding(88000, 16),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # model.summary()

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        x_val = train_data[:10000]
        x_train = train_data[10000:]

        y_val = train_labels[:10000]
        y_train = train_labels[10000:]

        fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
        
        results = model.evaluate(test_data, test_labels)
        print(results[1])
        if results[1] > best:
            best = results[1]
            model.save('model.h5')
else:
    print('file exists')
    model = keras.models.load_model('model.h5')

results = model.evaluate(test_data, test_labels)
print(results)

def review_encode(text):
    encoded = [1]
    for word in text:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

with open('text.txt', encoding='utf-8') as f:
    for line in f.readlines():
        # Convert the words to a number in a dictionary
        nline = line.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace(':', '').replace('[', '').replace(']', '').replace(';', '').replace(';', '').replace('\'', '').replace('\"', '').strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding='post', maxlen=250)
        pred = model.predict(encode)
        print('Original: {}\nNew: {}\nPrediction: {}'.format(line, encode, pred[0]))




"""# TEST
test_review = test_data[0]
predict = model.predict([test_review])
print('Review: {} \nPrediction: {} \nActual: {}'.format(decode_review(test_review), str(predict[0]), str(test_labels[0])))
"""