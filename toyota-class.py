# https://github.com/khuyentran1401/dog_classifier/blob/main/dog_classifier.py
from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st

st.write("""
# Toyota Model Classifier

Upload image of a Toyota car 

""")

class Predict:
    @st.cache(allow_output_mutation=True, max_entries=10, ttl=3600)
    def __init__(self, filename):
        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None
    
    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'Prediction: Toyota {pred}; Probability: {probs[pred_idx]:.04f}')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':

    file_name='model_resnet101.pkl'

    predictor = Predict(file_name)