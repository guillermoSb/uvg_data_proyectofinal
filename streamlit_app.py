import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
from models import ConvNet, infer
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image


"""
Universidad Del Valle De Guatemala
Data Science - 2023
"""
uploaded_file = st.file_uploader("Elegir un archivo", type=['png', 'jpg'])

# if uploaded_file is not None, process the image and show the result.
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # Create PIL image BGR->RGB
    pil_im = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    # Display the image
    st.image(pil_im, caption='Uploaded Image.', use_column_width=True)
    

    # Load model 01

    model = ConvNet(input_chanels=3, image_size=128).to('cpu')
    model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
    model.eval()

    prediction = infer(model,pil_im,)
    print("Predicted extent:", prediction)

    # Display the result on the streamlit page
    st.write("Predicted extent:", prediction)

    # Load model 02
   
    model_path = 'weighted_categories_model.h5'
    loaded_model = tf.keras.models.load_model(model_path)
    w = 64
    h = 64
    img = tf.expand_dims(cv2.resize(cv2_img, (w,h)), axis=0)
    prediction = loaded_model.predict(img)
    print("Predicted extent:", prediction[0][0])
    st.write("Predicted extent:", prediction[0][0] * 10)
