import os
import cv2
import numpy as np
from tqdm import tqdm
import streamlit as st
from PIL import Image
import pickle
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from numpy.linalg import norm
import tensorflow.keras.backend as K
K.clear_session()


# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Define the extract_feature function
def extract_feature(image_data, model):
    try:
        img = Image.open(image_data)
        img = img.convert("RGB")  # Convert to RGB format if necessary
        img = img.resize((224, 224))  # Resize to match ResNet50 input size
        img = np.array(img)
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)  # Use np.linalg.norm for calculating norm
        return normalized
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
        return None

# Define data directories for different categories
boys_directory = "data/boys"
girls_directory = "data/girls"

# Streamlit app
st.set_page_config(page_title="Apparel Recommendation System", page_icon=":shirt:", layout="wide")
st.title('Apparel Recommendation System')

# File upload section
uploaded_file = st.file_uploader("Upload an image of the apparel you like", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Display uploaded image
        st.sidebar.subheader("Uploaded Image")
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((224, 224))  # Resize image to match model input shape
        st.sidebar.image(resized_img, caption='Uploaded Image', use_column_width=True)

        # Feature extraction
        features = extract_feature(uploaded_file, model)

        if features is None:
            st.error("Error: Failed to extract features from the uploaded image.")
        else:
            # Recommendation checkboxes
            recommend_boys = st.sidebar.checkbox("Recommend for Boys")
            recommend_girls = st.sidebar.checkbox("Recommend for Girls")

            # Recommendation for boys' apparel
            if recommend_boys:
                st.subheader("Recommended Boys' Apparel:")
                boys_filenames = pickle.load(open('boys_filenames.pkl', 'rb'))
                boys_features = pickle.load(open('boys_features.pkl', 'rb'))
                nn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
                nn_model.fit(boys_features)
                distances, indices = nn_model.kneighbors([features])
                for index in indices[0][1:]:
                    st.image(Image.open(boys_filenames[index]), use_column_width=True)

            # Recommendation for girls' apparel
            if recommend_girls:
                st.subheader("Recommended Girls' Apparel:")
                girls_filenames = pickle.load(open('girls_filenames.pkl', 'rb'))
                girls_features = pickle.load(open('girls_features.pkl', 'rb'))
                nn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
                nn_model.fit(girls_features)
                distances, indices = nn_model.kneighbors([features])
                for index in indices[0][1:]:
                    st.image(Image.open(girls_filenames[index]), use_column_width=True)

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
