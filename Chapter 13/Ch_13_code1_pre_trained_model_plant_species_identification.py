
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import applications
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics

# Initialize training data lists
x_train = []
y_train = []

# Load and preprocess the images
for i in os.listdir('/content/drive/My Drive/DL/train/'):
    for j in os.listdir(f'/content/drive/My Drive/DL/train/{i}'):
        try:
            dummy = cv2.imread(f'/content/drive/My Drive/DL/train/{i}/{j}')
            dummy = cv2.resize(dummy, (128, 128))
            x_train.append(dummy)
            y_train.append(i)
        except Exception as e:
            print(e)

x_train = np.array(x_train)

# Encode the labels
dum = pd.get_dummies(y_train)
y_train = np.array(dum)

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

# Normalize the data
x_train = x_train / 255.0
x_val = x_val / 255.0

# Load VGG16 model
model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Freeze layers
for layer in model.layers[:11]:
    layer.trainable = False

# Add custom layers
cnn_out = model.output
ip_feat = Flatten()(cnn_out)
HL1 = Dense(1024, activation="relu")(ip_feat)
DO1 = Dropout(0.3)(HL1)
HL2 = Dense(1024, activation="relu")(DO1)
HL3 = Dense(64, activation="relu")(HL2)
predictions = Dense(11, activation="softmax")(HL3)

# Create the final model
model_final = Model(model.input, predictions)

# Compile the model
model_final.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])

# Define callbacks
checkpoint = ModelCheckpoint("/content/drive/My Drive/DL/plant_vgg16_best.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

# Train the model
epochs = 50
model_final.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks=[checkpoint, early])

# Predict on validation data
y_pred = model_final.predict(x_val)

# Convert predictions and true labels to class indices
y_label = [np.argmax(pred) for pred in y_pred]
y_actual = [np.argmax(true) for true in y_val]

# Print classification report and confusion matrix
cr = metrics.classification_report(y_actual, y_label)
print(cr)
cm = metrics.confusion_matrix(y_actual, y_label)
print(cm)
