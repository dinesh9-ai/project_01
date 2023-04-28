import streamlit as st 
from PIL import Image
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor



def upload_image():
    """Here we create a upload button for dropping the image"""

    img=None
    while not img:
        img=st.file_uploader("upload here")
    return img

def drop_down():
    count=st.selectbox('How many captions you want to generate',(1,2,3))
    return count

def load_models():
    """Here we load models"""

    model = VisionEncoderDecoderModel.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning").to("cpu")
    #print('loaded model')
    tokenizer = GPT2TokenizerFast.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning")
    #print('loaded tokenizer')
    image_processor = ViTImageProcessor.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning")
    #print('loaded image procssor')
    return [model,tokenizer,image_processor]

    
def run_app():
    """This function is the access point of the app"""

    with st.spinner("please wait model is loading"):
        [model,tokenizer,image_processor]=load_models()
        break
    st.empty()
    image=upload_image()
    caption=get_caption(model, image_processor, tokenizer, image)
    st.image(image)
    display_caption(caption)


def get_caption(model,image_processor,tokenizer,image):
    """In this function we take parmateres like model,image processor, tokenizer"""

    img=image_processor(image,return_tensors="pt").to("cpu")
    output=model.generate(**img)
    caption=tokenizer.batch_decode(ouput,skip_special_tokens=True)[0]
    return caption

def display_caption(caption):
    """This function helps in displaying the caption on screen"""
    
    st.markdown(caption)


run_app()