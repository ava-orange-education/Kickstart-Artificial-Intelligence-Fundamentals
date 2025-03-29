
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.utils import to_categorical 
from sklearn.metrics import classification_report
from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

# Represent Training & Testing samples suitable for TensorFlow backend
x_train = xtrain.reshape(xtrain.shape[0], 28, 28, 1).astype('float32')
x_test = xtest.reshape(xtest.shape[0], 28, 28, 1).astype('float32')

# Encoding the output class label (One-Hot Encoding)
y_train = to_categorical(ytrain, 10)
y_test = to_categorical(ytest, 10)

# Model Building
model = Sequential()
model.add(Conv2D(8, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile and Train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=500, epochs=100, verbose=1, validation_data=(x_test, y_test))

# Predictions and classification report 
y_pred = model.predict(x_test) 
y_pred_classes = y_pred.argmax(axis=1) 
print(classification_report(y_test.argmax(axis=1), y_pred_classes))
