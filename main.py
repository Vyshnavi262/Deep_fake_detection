from os import path, remove
from random import randint
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model import get_model, get_model_summary, model_prediction

# Function for face segmentation
def identify_fake_parts(image_path, threshold):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fake_mask = np.zeros_like(binary)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < threshold:
            cv2.drawContours(fake_mask, [contour], -1, (255), thickness=cv2.FILLED)
    return fake_mask

# Streamlit app
st.set_page_config(page_title="Real Fake Face Classification",
                   page_icon=path.join('static', 'icons', 'logo.png'))

with open(path.join('static', 'styles.css')) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

_, page_banner_img, _ = st.columns([3, 2, 3])
page_banner_img.image(path.join('static', 'icons', 'robot_face.png'),
                      use_column_width=True)

st.title('Real Fake Face Image Classifier')
st.subheader('')

st.markdown(
    """
    This real-fake face classifier utilizes a DenseNet-based convolutional neural network. Trained on a Kaggle dataset comprising 140,000 images (70,000 real and 70,000 fake), the classifier achieves an outstanding accuracy of 99%. Explore the model and performance metrics. The web application enables face classification with associated probabilities for the `Real` or `Fake` class.
"""
)

finalmodel = get_model(path.join('static', 'finalmodel', 'spoffnet.h5'))
model_summary = get_model_summary(finalmodel)

uploaded_file = st.file_uploader('Upload a face image for prediction',
                                 type=['png', 'jpg', 'jpeg'],
                                 accept_multiple_files=False,
                                 help='Please select an image for prediction')

threshold = st.slider('Segmentation Threshold', min_value=100, max_value=5000, value=1000)

if uploaded_file is not None and uploaded_file.type.split('/')[0] == 'image':

    uploaded_file_name = uploaded_file.name
    uploaded_image = Image.open(uploaded_file).save(uploaded_file_name)
    
    # Integrate face segmentation
    fake_mask = identify_fake_parts(uploaded_file_name, threshold)

    prediction_label, real_face_prob, fake_face_prob = model_prediction(uploaded_file_name, finalmodel)
    remove(uploaded_file_name)  # scrap the file after prediction

else:

    embedded_img = path.join('static', 'faces', f'{randint(1, 6)}.jpg')
    st.image(Image.open(embedded_img).resize((720, 720)))
    
    # Integrate face segmentation
    fake_mask = identify_fake_parts(embedded_img, threshold)
    

    prediction_label, real_face_prob, fake_face_prob = model_prediction(embedded_img, finalmodel)

if(prediction_label == 'Fake Face'):
    with st.expander(label='Segmented Image', expanded=False):
        st.image(fake_mask, caption='Identified Fake Parts (Expanded)', use_column_width=True)
with st.expander(label='Prediction Probabilities', expanded=False):
    st.markdown('<h3>Results</h3>', unsafe_allow_html=True)

    row_11, row_12 = st.columns([1, 1])
    row_11.markdown('<h4>Face Type</h4>', unsafe_allow_html=True)
    row_12.markdown('<h5>Probability</h5>', unsafe_allow_html=True)
    st.progress(0)

    row_21, row_22 = st.columns([1, 1])
    row_21.markdown('<h3>REAL</h3>', unsafe_allow_html=True)
    row_22.markdown(f'<h2>{real_face_prob} %</h2>', unsafe_allow_html=True)
    st.progress(real_face_prob)

    row_31, row_32 = st.columns([1, 1])
    row_31.markdown('<h3>FAKE</h3>', unsafe_allow_html=True)
    row_32.markdown(f'<h2>{fake_face_prob} %</h2>', unsafe_allow_html=True)
    st.progress(fake_face_prob)



st.markdown(f"The classifier's prediction is that the loaded image is a ```{prediction_label}```❗")

