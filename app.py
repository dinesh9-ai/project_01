import streamlit as st 
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv.main import load_dotenv
import os




def upload_image():
    """Here we create a upload button for dropping the image"""

    img=st.file_uploader("Upload the picture here",key="1")
    return img

def drop_down():
    count=st.selectbox('How many captions you want to generate',(None,1,2,3))
    return count

def authentication():
    """ Here we perform authentication for Azure-Computer Vision API"""

    load_dotenv()
    key=os.environ['key']
    endpoint=os.environ['endpoint']
    credentials=CognitiveServicesCredentials(key)
    client=ComputerVisionClient(endpoint=endpoint, credentials=credentials)
    return client

def caption_image(client,image,max_descriptions):
    """Here we get Azure CV API gives us the caption of our uploaded image"""

    analysis=client.describe_image_in_stream(image,max_descriptions,language="en")
    return analysis

def display_caption(analysis,count):
    """This function helps in displaying the caption on screen"""

    for i in analysis.captions:
        st.subheader(i.text)
    

def run_app():
    """This function is the access point of the app"""

    st.header("Image Captioning App")
    with st.spinner("please wait model is loading"):
        client=authentication()
        
    image=upload_image()
    count=drop_down()
    if image is not None and count is not None:
        analysis=caption_image(client, image,count)
        st.image(image)
        display_caption(analysis,count)



run_app()