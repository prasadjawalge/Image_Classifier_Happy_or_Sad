import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
st.title('Image Classifier Happy or Sad')

upload_file = st.sidebar.file_uploader('Upload image', type='jpg')
generate_predict = st.sidebar.button('Predict')
model = tf.keras.models.load_model('Happy_Sad_Model.h5')
def prediction(image, model):
    return model.predict(np.expand_dims(tf.image.resize(image, (256,256))/225, 0))

if generate_predict:
    image = Image.open(upload_file)
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = prediction(image, model)
    
    if pred > 0.5:
        st.text('Predicted image is Sad')
    else:
        st.text('Predicted image is Happy')
