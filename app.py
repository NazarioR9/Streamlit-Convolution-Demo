import os
import streamlit as st
import pandas as pd
import numpy as np
from convolution import my_convolution, read_for_conv


@st.cache
def list_images(path=IMG_PATH, join=False):
	imgs = os.listdir(path)
	return imgs if not join else [path+img for img in imgs]


def file_selector():
	selected = st.selectbox('Select an image', list_images())
	return IMG_PATH+selected

#************************

IMG_PATH = 'images/'

st.title('Convolution demos')
st.header('Images list')

st.image(image=list_images(join=True), caption=list_images(), width=100)

selected = file_selector()

raw_holder, conv_holder = st.beta_columns(2)

with raw_holder:
	st.image(selected)
	st.text('Selected image')

with conv_holder:
	st.image(my_convolution(read_for_conv(selected)), clamp=True)
	st.text('Convolved image')
