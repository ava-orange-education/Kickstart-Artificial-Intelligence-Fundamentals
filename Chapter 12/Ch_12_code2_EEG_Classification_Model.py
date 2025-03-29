
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Loading the dataset
path1 = "/DL/alcoholic"
path2 = "/DL/control"
files1 = os.listdir(path1)
num_samples1 = len(files1)

# Extracting the sensor value for alcoholic samples
dat_alcoholic = np.zeros((num_samples1, 16384))
for i, file in enumerate(files1):
    eeg = pd.read_csv(path1 + '/' + file)
    dat_alcoholic[i, :] = eeg['sensor value']

# Extracting the sensor value for control samples
files2 = os.listdir(path2)
num_samples2 = len(files2)
dat_control = np.zeros((num_samples2, 16384))
for i, file in enumerate(files2):
    eeg = pd.read_csv(path2 + '/' + file)
    dat_control[i, :] = eeg['sensor value']

# Creating labels
num_samples = num_samples1 + num_samples2
label = np.zeros((num_samples,), dtype=int)
label[:num_samples1] = 1  # Alcoholic
label[num_samples1:] = 0  # Controls

# Concatenating and shuffling data
dat_complete = np.concatenate((dat_alcoholic, dat_control), axis=0)
data, Label = shuffle(dat_complete, label, random_state=2)

# Splitting the data
xtrain, xtest, ytrain, ytest = train_test_split(data, Label, test_size=0.3, random_state=4)
x_train = xtrain.astype('float32').reshape(xtrain.shape[0], xtrain.shape[1], 1)
x_test = xtest.astype('float32').reshape(xtest.shape[0], xtest.shape[1], 1)
y_train = to_categorical(ytrain)
y_test = to_categorical(ytest)

# Defining the model
model = Sequential()
model.add(Conv1D(12, kernel_size=3, input_shape=(x_train.shape[1], 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))  # Assuming binary classification

# Model prediction
y_predict = model.predict(x_test)
y_pred = [np.argmax(val) for val in y_predict]

# Generating metrics
cr = metrics.classification_report(ytest, y_pred)
cm = metrics.confusion_matrix(ytest, y_pred)

print("Classification Report:\n", cr)
print("Confusion Matrix:\n", cm)
