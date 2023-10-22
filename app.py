import time
import streamlit as st
import pickle
from PIL import Image
from io import BytesIO
from PIL import Image as PILImage

from utils import eye_img_data, gaze_data, get_tracker

st.markdown(
     f"""
     <style>
     
    .stApp {{
            background: url("https://i.imgur.com/woiEqDB.png");
            background-size: cover
        }}

    .css-1n76uvr {{
        background-color: rgba(92, 65, 106, 0.6);
        padding: 30px;
        border-radius: 30px
    }}
    
    *{{
        color: white;
        text-align: center
    }}
    
    .css-1v0mbdj {{
        display: block;
        margin: 0 auto;
        width: 100px;
        margin-bottom: -50px
    }}
    
    .css-1offfwp{{
        padding-left: 5%;
        padding-right: 5%
    }}
 
     </style>
     """,
     unsafe_allow_html=True
 )

# loading

with open("eye_model.pkl", 'rb') as file:
    clf = pickle.load(file)
    
ready = False
    
with st.container():
    
    image = Image.open('eyeball.png')
    st.image(image)

    st.header('Innovsion')
    st.write("Whether you're in a work meeting on Zoom or rewatching lectures on Panopto, staying attentive online is hard. At Innovision, it's easy to gauge your attention levels based on eye tracking technology. ")
    
    tracker = get_tracker()
    data = eye_img_data(tracker)
    
    if not ready:

        st.subheader('1. Make sure we can see you!')
        st.write("Click the READY button once you're in position.")
        
        placeholder_image = PILImage.new("RGB", (640, 480))
        image_placeholder = st.image(placeholder_image, caption="Eye Image")
        
        ready = st.button('Ready')

        while True:
            
            # update image
            data = eye_img_data(tracker)
            image_io = BytesIO(data['image_data'])
            image = PILImage.open(image_io)
            image_placeholder.image(image, caption="Eye Image")
            time.sleep(0.1)
            
            if ready:
                break
            
    
    