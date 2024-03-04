import streamlit as st
st.title("Fashion Recommender")
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open("C:\\Users\\Ananya Chaturvedi\\PycharmProjects\\fashionRecommenderSystem\\images\\embeddings.pkl", 'rb')))
filenames = pickle.load(open("C:\\Users\\Ananya Chaturvedi\\PycharmProjects\\fashionRecommenderSystem\\images\\filenames.pkl", 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')


def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        print(f"Saving file to {file_path}")
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        print(f"Error saving file: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # displays the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # extraction of features
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # st.text(features)
        # recommendations
        indices = recommend(features, feature_list)
        # show
        col1, col2, col3, col4, col5 = st.beta_columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occurred while uploading the file")