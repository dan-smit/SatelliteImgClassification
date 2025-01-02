import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.manifold import Isomap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from dsmith728_OneClassClassification import getImageMatrix, getImageLabel

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
# print(f'y_train.shape: {y_train.shape}') #(700,)

#Obtain vectorized representation for each image, stored in a matrix
X_train = getImageMatrix(train_images_path)  #(700, 1228800)
X_test = getImageMatrix(test_images_path) #(100, 1228800)

#standardize data
X_train = X_train / 255
X_test = X_test / 255

#reproducible results
np.random.seed(7)

#Acquire data subsets for hyperparameter tuning.
s_split = StratifiedShuffleSplit(n_splits=3, test_size=0.75, random_state=7)
for train_idx, _ in s_split.split(X_train, y_train):
    X_train_subset = X_train[train_idx]
    y_train_subset = y_train[train_idx]
    
# print(f'X_train_subset.shape: {X_train_subset.shape}') #(175, 1228800)

#SVM (rbf kernel)
#ISOMAP (nonlinear dimensionality reduction)
# Use a pipeline to tune hyperparameters. RandomizedSearchCV is used instead of GridSearch in order
# to reduce compute time. A subset of the training data is used for CV to reduce comp time as well.
# GridSearch tests all combinations of hyperparameters --> computationally expensive

#The best parameters turn out to be gamma = 0.001, C = 10, and isomap n_neighbors = 10
pipeline = Pipeline([
    ('isomap', Isomap()),
    ('svm', SVC(kernel='rbf'))
])

# print(pipeline.get_params().keys())

param_grid = {
    'isomap__n_neighbors': [5, 10, 15, 20, 25, 30],
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1, 10] 
}
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=25,
    cv=5,
    scoring='accuracy',
    random_state=7
)

random_search.fit(X_train_subset, y_train_subset)

print("Best parameters:", random_search.best_params_)


isomap = Isomap(n_neighbors = 10, n_components = 2)
X_train_isomap = isomap.fit_transform(X_train)
X_test_isomap = isomap.transform(X_test)
svm_rbf_model = SVC(kernel='rbf', C=10, gamma=0.001).fit(X_train_isomap, y_train)
svm_rbf_predictions = svm_rbf_model.predict(X_test_isomap)
svm_rbf_cm = confusion_matrix(y_test, svm_rbf_predictions)

svm_rbf_cm_disp = ConfusionMatrixDisplay(svm_rbf_cm)
svm_rbf_cm_disp.plot(colorbar = False)
plt.title(f'SVM Confusion Matrix')

plt.tight_layout()
plt.show()