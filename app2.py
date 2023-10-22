import threading
import time
import streamlit as st
import pickle
from PIL import Image
from io import BytesIO
from PIL import Image as PILImage
from flask import session

from utils import build_dataset, eye_img_data, gaze_data, get_gazepoints, get_tracker, process_dataframe

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
        padding-left: 10px;
        padding-right: 10px;
    }}
    
    .css-1x8cf1d.edgvbvh10{{
        padding: 0 px;
        padding-left: 10%;
        padding-right: 10%;
        background-color: black;
        margin-top: 20px
        
    }}
    
     </style>
     """,
     unsafe_allow_html=True
 )

# loading

with open("eye_model.pkl", 'rb') as file:
    clf = pickle.load(file)
    
record = False
    
with st.container():
    
    image = Image.open('eyeball.png')
    st.image(image)

    st.header('Innovsion')
    st.write("Whether you're in a work meeting on Zoom or rewatching lectures on Panopto, staying attentive online is hard. At Innovision, it's easy to gauge your attention levels based on eye tracking technology. ")
    
    tracker = get_tracker()
    data = gaze_data(tracker)
    
    st.subheader('1. Set up your tracking info')
    
    record_time = st.number_input('How many minutes do you want to record for?', value = 0.1)
    time_step = st.number_input('How many seconds between each recorded gaze-point?', value = 0.1)
    
    record = st.button('Ready')    
        
    if record:
        st.subheader('Tracking your eye movements...')
        df, _ = build_dataset(tracker, 'user', time_step_sec=time_step, tot_time_min=record_time)

        st.write(df)

        st.subheader('All done!')
        st.write('Hope your online meeting or lecture went well.')

        df = process_dataframe(df)
        df = get_gazepoints(df)
        df = get_gazepoints(df, "right")

        st.write(df)