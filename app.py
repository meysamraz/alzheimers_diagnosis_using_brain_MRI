from fastai.vision.all import *
from fastai.vision import *
import streamlit as st
from PIL import Image


st.markdown("""
        <style>
            .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
            .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)



st.title('Alzheimers diagnosis using brain MRI')

# Load pre traind model
learn_ff = load_learner('xresnet34_export.pkl', cpu=False)

st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload image (allowed jpg,jpeg and png)
file_up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if file_up is not None:
    image = Image.open(file_up)
    row1,row2,row3,row4 = st.columns(4)
    row1.write("")
    row2.image(image, caption='Uploaded your.', width=380)
    row3.write("")
    st.write("")
    labels = learn_ff.predict(np.array(image))[0]
    # print result
    st.write("Result : ",labels)