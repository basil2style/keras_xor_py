
# coding: utf-8

# In[1]:

#Following : https://blog.thoughtram.io/machine-learning/2016/09/23/beginning-ml-with-keras-and-tensorflow.html
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense


# In[2]:

# Four different states of XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]],"float32")


# In[4]:

# Four expected results in the same order
target_data = np.array([[0],[1],[1],[0]],"float32")


# In[5]:

model = Sequential()


# In[6]:

model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[7]:

# 'binary_accuracy', which tells how good the predictions of our NN after each epoch (.75(75%) = 1(100%))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])


# In[11]:

#Training a neural network happens in iterations so called epochs.
model.fit(training_data, target_data, nb_epoch=10, verbose=2)


# In[9]:

print model.predict(training_data).round()


# In[ ]:



