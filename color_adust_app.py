from matplotlib import pyplot as plt
import cv2
import numpy as np

import torch
import kornia
import torchvision

from PIL import Image

import streamlit as st

DEMO_IMAGE = 'kids.jpg'




def imshow(input: torch.Tensor):
        
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = kornia.tensor_to_image(out)
        
    st.image(out_np,use_column_width = True,channels = 'BGR')



    
    
    #title
st.title('Color Adjustment using Kornia')
    
st.sidebar.title('Color Adjustment using Kornia')
    
st.sidebar.subheader('Parameters')
    
    #input file
    
uploaded_file = st.file_uploader("Upload an Image",type=[ "jpg", "jpeg"])
if uploaded_file is not None:
    im = np.array(Image.open(uploaded_file))
else:
    demo_image= DEMO_IMAGE
    im = np.array(Image.open(demo_image))
        
        
   
im = np.array(im)
#im = im[:,:,::3]    

    
    
    
img_bgr: np.ndarray = im
    
st.text('Original Image')
    
st.image(img_bgr,channels='RGB',use_column_width=True)
    
    #############################
# Convert the numpy array to torch
x_bgr: torch.Tensor = kornia.image_to_tensor(img_bgr)
x_rgb: torch.Tensor = kornia.bgr_to_rgb(x_bgr)

#############################
# Create batch and normalize
x_rgb = x_rgb.expand(4, -1, -1, -1)  # 4xCxHxW
x_rgb = x_rgb.float() / 255.
    
    
    
    
    
    
    
st.subheader('Brightness Image')
    
st.sidebar.text('Brightness Parmeters')
brightness = st.sidebar.slider('Brightness',min_value= 0.0 , max_value=1.0,value = 0.6)
brightness = np.clip(brightness,0.0,1.0)
    
    #brightness 
x_brightness: torch.Tensor = kornia.enhance.adjust_brightness(x_rgb,brightness)
    
imshow(x_brightness)
    
st.subheader('Contrast Image')
    
st.sidebar.text('Contrast Parmeters')
contrast = st.sidebar.slider('Contrast',min_value= 0.0 , max_value=1.0,value = 0.2)
contrast = np.clip(contrast,0.0,1.0)
    
x_contrast: torch.Tensor = kornia.enhance.adjust_contrast(x_rgb,contrast)
imshow(x_contrast)
    
    # Adjust Gamma
    
st.subheader('Gamma Image')
    
st.sidebar.text('Gamma Parmeters')
gamma1 = st.sidebar.slider('Gamma',min_value= 1.0, max_value=10.0,value = 3.0)
gamma1 = np.clip(gamma1,1.0,10.0)
gain1 = st.sidebar.slider('Gain',min_value= 1.0, max_value=10.0,value = 1.5)
gain1 = np.clip(gain1,1.0,10.0)


x_gamma: torch.Tensor = kornia.enhance.adjust_gamma(x_rgb, gamma=gamma1, gain=gain1)

imshow(x_gamma)
    
    # Adjust Saturation
st.subheader('Saturation Image')
st.sidebar.text('Saturation Parmeters')
saturation = st.sidebar.slider('Saturation',min_value= 0.0 , max_value=1.0,value = 0.2)
saturation = np.clip(saturation,0.0,1.0)
x_saturated: torch.Tensor = kornia.enhance.adjust_saturation(x_rgb, saturation)
imshow(x_saturated)
    
    # Adjust Hue
st.subheader('Hue Image')
st.sidebar.text('Hue Parmeters')
hue = st.sidebar.slider('Hue',min_value= 0.0 , max_value=1.0,value = 0.5)
hue = np.clip(hue,0.0,1.0)
x_hue: torch.Tensor = kornia.enhance.adjust_hue(x_rgb, hue)
imshow(x_hue)

st.markdown(
    """
    # Created using Kornia
    
    * You can install **Kornia** using pip or pip3
    
    ```
    pip install --upgrade kornia
    
            (or)
            
    pip3 install --upgrade kornia
    
    ```
           
    * You can check the documentation from [here](https://kornia.readthedocs.io/en/latest/)
    
    * Created by **[Pavan Kunchala](https://www.linkedin.com/in/pavan-kumar-reddy-kunchala/)**

    * I am also looking for Freelancing and Collabration Opportunites in the field of **Deep Learning** and **Computer Vision**,if you are willing collaborate to mail me at (pavankunchalapk@gmail.com)
    
    """
)


    
    

    
    
    
    
