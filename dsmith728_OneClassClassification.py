
# The purpose of this file is to perform necessary pre-processing steps in order to acquire one class classification (OOC)
# labels for satellite images. It does this by using the label with the biggest bounding box.
# In addition, this file will vectorize each image, and format it into a matrix for model training

from PIL import Image
import os
import numpy as np

#unflattenedImages vectorizes each image into a matrix. Each row is the image. Each image has shape (image_len, image_len, 3)
def unflattenedImages(images_path, image_len):
    data_arrays = []
    for filename in os.listdir(images_path): 
        img_path = os.path.join(images_path, filename)
        # print(f'img_path: {img_path}')
        img = Image.open(img_path).convert('RGB')
        img = img.resize((image_len, image_len))
        img_array = np.array(img)
        # print(f'img_array.shape: {img_array.shape}') #(256 x 256 x 3)
        data_arrays.append(img)
    
    images_matrix = np.array(data_arrays)
    return images_matrix

#The getImageMatrix function is the same thing as unflattenedImages, except its strictly 640x640
def getImageMatrix(images_path):
    data_arrays = []
    for filename in os.listdir(images_path): 
        img_path = os.path.join(images_path, filename)
        # print(f'img_path: {img_path}')
        img = Image.open(img_path).convert('RGB')
        img = img.resize((640, 640))
        img_array = np.array(img).flatten()
        # print(f'img_array.shape: {img_array.shape}') #(1228800,) = (640 x 640 x 3)
        data_arrays.append(img_array)
    
    images_matrix = np.array(data_arrays)
    return images_matrix

#getImageLabel assigns the object with the biggest bounding box as the image label

def getImageLabel(labels_path):
    labels = []
    for filename in os.listdir(labels_path):
        txt_path = os.path.join(labels_path, filename)
        with open(txt_path, 'r') as f:
            clean_arr = []
            for line in f:
                arr = organizeArrayObject(line.split())
                clean_arr.append(arr)
            obj_id = getBiggestObject(clean_arr)
            labels.append(obj_id)
        f.close()
    return labels

def organizeArrayObject(arr):
    x = []
    x.append(int(arr[0]))
    for i in range(1, len(arr)):
        x.append(float(arr[i]))
    return x

def getBiggestObject(arrays):
    object_id_areas = {}

    for obj in arrays:
        area = obj[3] * obj[4]
        if obj[0] in object_id_areas:
            object_id_areas[obj[0]] += area
        else:
            object_id_areas[obj[0]] = area
    max_object_id = max(object_id_areas, key=object_id_areas.get)
    return max_object_id