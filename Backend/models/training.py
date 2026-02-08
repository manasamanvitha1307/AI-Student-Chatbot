import tensorflow
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import json
import pickle

words = []
labels = []
docs = []
ignore_list = ['?', '!']

dataset = open('intents.json').read()
intents = json.loads(dataset)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_token = nltk.word_tokenize(pattern)
        words.extend(word_token)
        docs.append((word_token, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_list]
words = sorted(list(set(words)))
labels = sorted(list(set(labels)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

training_data = []
output = [0] * len(labels)

for doc in docs:
    bag_of_words = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag_of_words.append(1 if w in pattern_words else 0)

    output_row = list(output)
    output_row[labels.index(doc[1])] = 1

    training_data.append([bag_of_words, output_row])

random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)
x_train = np.array([item[0] for item in training_data])
y_train = np.array([item[1] for item in training_data])

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])


history = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)

model.save('chatbot_Application_model.h5', history)