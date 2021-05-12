# Importing the libraries

import numpy as np
import pandas as pd
import tensorflow as tf


# Data pre-processing       [Part - 1]
    # Importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


    # Encoding the categorical data
        # Label encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

        # One hot encoding

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = ct.fit_transform(x)


# Splitting the dataset into train and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# feature scaling       [In deep learning feature scaling should be applied to the whole features set]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Building an ANN        [Part - 2]

    # Initialising an ANN

ann = tf.keras.models.Sequential()

    # Adding an input layer and first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Training the ANN      [Part - 3]

    # Compiling the ANN

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the ANN

ann.fit(x_train, y_train, batch_size=32, epochs=50)


# Predicting the results and evaluating the model

    # making the prediction for single observation
# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))     # This is probability results

print('Does this person leave the bank?', (ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5))   # This is boolean results


    # Making the predictions for test set
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

    # displaying the concatenation

concat = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

print(concat)

    # Making the confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))
