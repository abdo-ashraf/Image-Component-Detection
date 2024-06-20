## import required libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import pandas as pd


def load_image(image_file):
    """
    Loads an image from the given file.
    
    Parameters:
    image_file (UploadedFile): The image file uploaded by the user.
    
    Returns:
    Image: An Image object representing the loaded image.
    """
    img = Image.open(image_file)
    return img


def classify_image(image):
    """
    Classifies the given image into different categories using a pre-trained MobileNetV2 model.
    The function resizes the image, normalizes it, and predicts the top 5 components present in the image.
    
    Parameters:
    image (Image): An Image object to be classified.
    
    Returns:
    dict: A dictionary with two keys 'Components' and 'Confidence'. 
          'Components' contains a list of 5 detected component names.
          'Confidence' contains a list of confidence scores for each detected component.
    """
    # Dictionary to hold labels and predictions
    lbl_dict = {'Components': [],
                'Confidence': []}
    
    # Ensure the image is in RGB format
    image = image.convert("RGB")
    
    # Resize the image to the required input size for MobileNetV2
    image = image.resize((224, 224))
    
    # Normalize the image array to have values between 0 and 1
    image_array = np.array(image) / 255.0
    
    # Expand dimensions to match the input shape expected (Batch, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make predictions using the pre-trained MobileNetV2 model
    predictions = MODEL.predict(image_array)
    
    # Decode the top 5 predictions
    top_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    
    # Append top-5 predictions to the dictionary
    for pred in top_predictions:
        lbl_dict['Components'].append(pred[1])
        lbl_dict['Confidence'].append(pred[2])
    
    # Return dictionary of the component names and confidence scores
    return lbl_dict

# Load a pre-trained object detection model
MODEL = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
LABELS_PATH = tf.keras.utils.get_file('imagenet_class_index.json', 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')


with open(LABELS_PATH, 'r') as f:
    LABELS = json.load(f)


# Streamlit app layout
st.title("Image Component Detection")
st.write("Upload an image to detect its components")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Analyse Image'):
        st.write("Analyzing...")
        dict_predictions = classify_image(image)
        
        st.write("Detected Components:")
        # for component, score in predictions:
        #     st.write(f"{component}: {score:.2f}")
        st.write(pd.DataFrame(dict_predictions, index=('Top-1','Top-2','Top-3','Top-4','Top-5')))
