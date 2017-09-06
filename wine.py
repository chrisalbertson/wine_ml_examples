#!/usr/bin/env python
'''
This is a purposefully simple deep learning program inteded t searve as a "Hello World" example

It takes as input a standard dataset of chemical analysis of wine samples and a one to ten
star rating by a human "wine expert".  The softwaretries to predict the rating from the chemical
analisys.   The example network is NOT a clasiifier as the output is a continous floating
point number.

The models results are not perfect.  For from it.  One ofthe lessons this example teaches is
that the results are only as good as your trainning data.   Chemical analysis, it turns out
is not a great predictor of wine quality.  Also human "experts" are not as good at assingning
a rating as they woul like to think.

ON the other hand the other example program wine_simple.py makes veery accurate preductions because
it is infact possible to determine red from white wine by chemical analysis.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
import numpy as np
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Some parameters
#task = 'get_type'
task = 'get_quality'

# Read the two data files.
red   = pd.read_csv("winequality-red.csv",   sep=';')
white = pd.read_csv("winequality-white.csv", sep=';')

# Combine the two files into one
red["type"]   = 1
white["type"] = 0

wine =  red.append(white, ignore_index=True)


if task == 'get_type':

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

    # Create the model, compile it then print a summary
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(11,)))
    model.add(Dense(8,  activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
    model.add(Dense(6,  activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
    model.add(Dense(1,  activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    # Train the model
    model.fit(x_train, y_train, epochs=20, batch_size=4, verbose=1)

    # Test the model and print the score
    score = model.evaluate(x_test, y_test,verbose=1)
    print(score)

elif task == 'get_quality':

    # Split the combined data into X and Y and training and test data sets
    print('wine.shape =', wine.shape)
    x = wine.drop('quality', axis=1)
    y = wine.quality
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)

    print('x_train.shape = ', x_train.shape ) ## TODO

    # Standarize the data (only the X needs this, Y is a scaler)
    s = StandardScaler().fit(x_train)    # Use stats from training set
    x_train = s.transform(x_train)
    x_test  = s.transform(x_test)

    # Create the model, compile it then print a summary
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=12))
    model.add(Dense(32, activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
    model.add(Dense(16, activation='relu',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
    model.add(Dense(1)) # no activation funtion.  We want a scaler output

    rmsprop = RMSprop(lr=0.000075)
    model.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])

    print(model.summary())

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1)

    # Test the model and print the score
    score = model.evaluate(x_test, y_test,verbose=1)
    print(score)

    # Finally we have a model we can use to make predictions
    out = model.predict(x_test, batch_size=8, verbose=1)

    # make a scatter plot of truth vs. prediction
    plt.scatter(y_test, out, color="white", edgecolors="red", lw=0.5)
    plt.show()

else:
    print('task is invalid')
