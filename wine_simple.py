#!/usr/bin/env python
'''
This is a purposefully simple deep learning program inteded t searve as a "Hello World" example

This example is purposly very simple and serves as a "hello world" kind of example for a
Keras based logistic classicier.

Of the two example networks this is the simplest one and the one that makes the most accurate predictions
It learns to classify wine as either red or while based on chamical analysis.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Read the two data files.
red   = pd.read_csv("winequality-red.csv",   sep=';')
white = pd.read_csv("winequality-white.csv", sep=';')

# Combine the two files into one
red["type"]   = 1
white["type"] = 0
wine =  red.append(white, ignore_index=True)

# Split the combined data into X and Y and trainning and test datesets
x=wine.ix[:,0:11]
y= np.ravel(wine.type)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.33,
                                                    random_state=42)

# Standarize the data (only the X needs this, Y is alread 0 and 1)
s = StandardScaler().fit(x_train)    # Use stats from training set
x_train = s.transform(x_train)
x_test  = s.transform(x_test)

# Create the model, compile it and print a summary
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(11,)))
model.add(Dense(8,  activation='relu'))
model.add(Dense(6,  activation='relu'))
model.add(Dense(1,  activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=1)

# Test the model and print the score
score = model.evaluate(x_test, y_test,verbose=1)
print(score)
