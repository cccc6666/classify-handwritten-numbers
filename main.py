#keras fully connected

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential;
from keras.layers import Convolution1D, Flatten, Dense, Dropout,MaxPooling1D, regularizers
from keras.utils import to_categorical
from keras import optimizers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
d = train.iloc[:,0]
train = train.iloc[:,1:]
test = test.iloc[:,:]
labels = to_categorical(d, num_classes=10)

network = Sequential()
network.add(Dense(64, input_dim = 784, activation='relu'))
network.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
network.add(Dropout(0.1))
network.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
network.add(Dropout(0.2))
network.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
network.add(Dropout(0.2))
network.add(Dense(10, activation = 'softmax'))
network.compile('Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(train, labels, epochs=30)
testoutput = network.predict(test)

predictions = []
for i in range(len(testoutput)):
    predictions.append(int(np.argmax(testoutput[i,:])))

predictions_file = pd.DataFrame(predictions)
predictions_file.index = np.arange(1,len(predictions_file)+1)
predictions_file.columns = ["Label"]
predictions_file.to_csv('predictions.csv')