import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


#Set variable = 0 to prevent error message. Fixes floating-point round-off errors.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from dsmith728_OneClassClassification import getImageMatrix, getImageLabel, unflattenedImages

#paths for training/validation/test images and labels
train_images_path = './data/train/images'
train_labels_path = './data/train/labels'

valid_images_path = './data/valid/images'
valid_labels_path = './data/valid/labels'

test_images_path = './data/test/images'
test_labels_path = './data/test/labels'

#Obtain training and test label through one-class classification
#In this case, OOC uses the label with the biggest bounding box for each image.

y_train = np.array(getImageLabel(train_labels_path)) #(700,)
y_test = np.array(getImageLabel(test_labels_path)) #(100,)
y_valid = np.array(getImageLabel(valid_labels_path)) #(200,)
# print(f'y_train.shape: {y_train.shape}') #(700,)

#Convert y_train and y_test into one-hot encoding for multi class classification. i.e, if y_0 == 7, the one-hot encoding
#equivalent would be [0 0 0 0 0 0 1 0 0 0 0 0 0 0] since there are 14 classes
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)
y_valid_onehot = to_categorical(y_valid)

print(f'y_train_onehot.shape: {y_train_onehot.shape}')
# print(f'y_train[0]: {y_train[0]}') #2
# print(f'y_train_onehot[0]: {y_train_onehot[0]}') #[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# print(f'y_train_onehot.shape: {y_train_onehot[0]}') #(700, 14)

#Obtain vectorized representation for each image, stored in a matrix
#256 x 256 images
img_len = 256
X_train = unflattenedImages(train_images_path, img_len) 
X_test = unflattenedImages(test_images_path, img_len) 
X_valid = unflattenedImages(valid_images_path, img_len)

#standardize data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_valid = X_valid.astype('float32') / 255

# print(f'X_train.shape: {X_train.shape}') #(700, 256, 256, 3)
# print(f'X_test.shape: {X_test.shape}') #(100, 256, 256, 3)
# print(f'X_valid.shape: {X_valid.shape}') #(200, 256, 256, 3)

#reproducible results
np.random.seed(7)

classes = np.unique(y_train)
nClasses = len(classes)
print(f'classes: {classes}')
print(f'nClasses')

#Define CNN model. For the convolution layers, the Rectified Linear Units (ReLu) activation function is used since it tends to be more effective
#than the logistic sigmoid function. ReLu helps the network learn non-linear decision boundaries.

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_len, img_len, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'), #added layer
    MaxPooling2D(pool_size=(2, 2)), #added layer
    Flatten(),
    Dense(256, activation='relu'),
    Dense(nClasses, activation='softmax')  # Multiclass classification
])

#Next we compile this model using Adam, a popular optimization algorithm used in computer vision. Adam is an extension to 
#stoachastic gradient descent
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# cnn_model.summary()

#Train the model using data subset 
cnn_train = cnn_model.fit(X_train, y_train_onehot, verbose=1, epochs = 10, validation_data=(X_valid, y_valid_onehot), batch_size=32)
# print(f'type cnn_train: {type(cnn_train)}') #<class 'keras.src.callbacks.history.History'>

#Evaluate performance
test_eval = cnn_model.evaluate(X_test, y_test_onehot)
print(f'Test loss: {test_eval[0]}')
print(f'Test Accuracy: {test_eval[1]}')

y_predictions = cnn_model.predict(X_test)
# print(y_predictions.shape) #(100, 14)

cnn_predictions = y_predictions.argmax(axis=1)
cnn_cm = confusion_matrix(y_test, cnn_predictions)
print(classification_report(y_test, cnn_predictions, zero_division=0.0))
cnn_cm_disp = ConfusionMatrixDisplay(cnn_cm)
cnn_cm_disp.plot(colorbar = False)
plt.title(f'CNN Confusion Matrix')

plt.tight_layout()
plt.show()

#Plot accuracy and loss plots
train_accuracy = cnn_train.history['accuracy']
val_accuracy = cnn_train.history['val_accuracy']
train_loss = cnn_train.history['loss']
val_loss = cnn_train.history['val_loss']

plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#Check class imbalance in training/validation/test data. Imbalanced data leads to model bias
train_classes, num_train_classes = np.unique(y_train, return_counts = True)
valid_classes, num_valid_classes = np.unique(y_valid, return_counts = True)
test_classes, num_test_classes = np.unique(y_test, return_counts = True)

x = np.arange(14) #label locations
width = 0.25 #bar widths

mapping = {0: 'Agricultural Sector', 1: 'Terminal', 2: 'Beach', 3: 'City', 4: 'Desert', 5: 'Forest', 6: 'Road', 7: 'Lake',
           8: 'Mountain', 9: 'Car Parking', 10: 'Port', 11: 'Train', 12: 'Domestic', 13: 'River'}

class_labels = [mapping[i] for i in range(len(mapping))]
bar_positions = np.arange(len(class_labels))

plt.bar(bar_positions - 0.25, num_train_classes, width=0.25, label='Training', color='orange')
plt.bar(bar_positions, num_valid_classes, width=0.25, label='Validation', color='blue')
plt.bar(bar_positions + 0.25, num_test_classes, width=0.25, label='Test', color='green')


plt.xlabel('Class Label')
plt.ylabel('Number of Data Points')
plt.title('Class Distributions')
plt.xticks(bar_positions, labels=class_labels, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()