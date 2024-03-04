import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from PIL import Image
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(file, model):
    image = Image.open(file)
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image_array = np.asarray(image)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_input = preprocess_input(expanded_image_array)
    features = model.predict(preprocessed_input)
    return features.flatten()

filenames = []
for file in os.listdir("C:\\Users\\Ananya Chaturvedi\\PycharmProjects\\fashionRecommenderSystem\\images"):
    filenames.append(os.path.join("C:\\Users\\Ananya Chaturvedi\\PycharmProjects\\fashionRecommenderSystem\\images", file))

print(len(filenames))
print(filenames[0:5])

feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

      #print(np.array(feature_list).shape)
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))