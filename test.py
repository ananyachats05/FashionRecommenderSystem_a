import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import GlobalMaxPooling2D
import pickle
from numpy.linalg import norm

from keras.src.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

model = ResNet50(weights='imagenet', include_top=False)
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

feature_list = np.array(pickle.load(
    open("C:\\Users\\Ananya Chaturvedi\\PycharmProjects\\fashionRecommenderSystem\\images\\embeddings.pkl", 'rb')))
filenames = pickle.load(
    open("C:\\Users\\Ananya Chaturvedi\\PycharmProjects\\fashionRecommenderSystem\\images\\filenames.pkl", 'rb'))

image = load_img(
    "C:\\Users\\Ananya Chaturvedi\\OneDrive\\Desktop\\sample\\saree.jpeg",
    target_size=(224, 224))
image = image.convert('RGB')
image_array = np.asarray(image)
expanded_image_array = np.expand_dims(image_array, axis=0)
preprocessed_input = preprocess_input(expanded_image_array)
features = model.predict(preprocessed_input)
result = features.flatten()
normalised_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(feature_list)

distance, indices = neighbors.kneighbors([normalised_result])

print(indices)
'''
for file in indices[0]:
    print(filenames[file])
    '''


for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)  # so that it does not disappear

