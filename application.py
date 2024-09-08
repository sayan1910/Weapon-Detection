# %%blocks



from PIL import Image
import tqdm
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import tensorflow as tf
from tensorflow.keras.preprocessing import image
# import torch
# import torchvision
from ultralytics import YOLO
import base64
import tempfile
import io
import cv2
import subprocess
# import wandb

# print(torch.__version__)
# print(torchvision.__version__)


import streamlit as st
import streamlit.components.v1 as components
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print('-'*50)
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # print(gpu)
print('-'*50)


# DOWNLOAD BUTTON============================================================
def download_button(object_to_download, download_filename, button_text, key):
    # Function to create a download button
    if isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
    elif isinstance(object_to_download, Image.Image):
        buffered = io.BytesIO()
        object_to_download.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode()
    else:
        raise ValueError("Unsupported object type for download_button")

    file = f"data:application/octet-stream;base64,{b64}"
    href = f'<span>Download result <a href="{file}" download="{download_filename}">{download_filename}</a></span>'
    st.markdown(href, unsafe_allow_html=True)
    # st.download_button(button_text, file, download_filename)


# THE MODEL============================================================
# @st.cache(max_entries=1, hash_funcs={"MyUnhashableClass": lambda _: None})
@st.cache_resource(max_entries=1, hash_funcs={"MyUnhashableClass": lambda _: None})
def get_model():
# wandb.login(key='9609b85b30dcd32e6169dda61daa747e6503bf8f')
    data_dir = "dataset.yaml"
    best_model_dir = "runs/detect/train/weights/best.pt"
    last_model_dir = "runs/detect/train/weights/last.pt"
    model = YOLO(best_model_dir)
    print('YOLO model loaded.')
    return model


# IMAGE INPUTS============================================================
testing = [
    'https://static01.nyt.com/images/2022/07/28/us/00arming-teachers01/00arming-teachers01-mediumSquareAt3X.jpg',
    'https://marvel-b1-cdn.bc0a.com/f00000000269380/www.beretta.com/en-us/assets/39/7/pistol1.jpg',
    'https://sources.roboflow.com/RN35QmVSLDW76eeVCpwwytGgVmn2/o2e1PTw3hpmdZsvh1SGO/original.jpg',
    'https://d3i71xaburhd42.cloudfront.net/e2a982da84bf199636f8cbc850d8979f99ce39c3/2-Figure1-1.png',
]
def preprocess_image(img_path, target_size=(130, 130)):
    # img = image.load_img(img_path, img_path)
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

def process_images(img, model):
    
    # preprocessing input image
    img = preprocess_image(img)[0]
    
    img = img.astype(np.uint8)
    img = image.img_to_array(Image.fromarray(img[..., ::-1]))
    result = model(img)[0]  # return a list of Results objects

    im_array = np.array(result.plot())  # plot a BGR numpy array of predictions
    im_array = im_array.astype(np.uint8)
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    return im


# VIDEO INPUTS============================================================
def compress_video(input_file, output_file):
    # Input and output file paths
    # input_file = os.path.join(f'/Users/jijnasu/Downloads/op-CQC-2s.mp4')
    # output_file = os.path.join(f'/Users/jijnasu/Downloads/op-CQC-2s--2.mp4')
    # print(input_file)
    # FFmpeg command for video compression
    ffmpeg_command = f'ffmpeg -i {input_file} -c:v libx264 -crf 23 {output_file}'

    # Run FFmpeg command
    res = subprocess.run(ffmpeg_command, shell=True)
    # print(res)

def process_frame(frame):
    im_array = frame.plot()
    im = Image.fromarray(im_array[..., ::-1])
    frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return frame

def preprocess_video(video, video_path):
    with open(video_path, "wb") as video_file:
        video_file.write(video.read())

    video_file = open(video_path, "rb")
    return video_file

def process_video(video, model):
    # preprocess video
    video_name = video.name.split('.')[0]
    video_path = f'{video_name}.mp4'
    video_name = ''.join(v if v.isalnum() else ' ' for v in video_name)
    video_name = video_name.replace(' ', '-')
    video = preprocess_video(video, video_path)

    st.write('Realtime detection')

    # model inferance
    # results = model(video_path, stream=True, device='mps')
    results = model(video_path, stream=True)

    # first frame
    first_result = next(results, None)
    if first_result is None:
        print("No results found.")
        exit()

    # Set video properties
    frame_height, frame_width = first_result.orig_shape

    # Process the video frames
    cap = cv2.VideoCapture(video_path)

    processed_video_path = f'optemp-{video_name}.mp4'
    processed_video_writer = cv2.VideoWriter(
        processed_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        25,
        (frame_width, frame_height)
    )
    # out = cv2.VideoWriter(f'{op_dir}_MJPG.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (frame_width, frame_height))
    # out = cv2.VideoWriter(f'{op_dir}_XVID.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (frame_width, frame_height))
    # out = cv2.VideoWriter(f'{op_dir}_H264.mp4', cv2.VideoWriter_fourcc(*'H264'), 25, (frame_width, frame_height)) #lowest file size
    # out = cv2.VideoWriter(f'{op_dir}_mp4v.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

    # Create an empty placeholder for the processed video
    processed_video_placeholder = st.empty()
    processed_frame = process_frame(first_result)
    processed_video_placeholder.image(processed_frame, channels="BGR")

    # Write the processed frame to the output video
    processed_video_writer.write(processed_frame)
    
    for i, result in enumerate(results):
        processed_frame = process_frame(result)
        processed_video_placeholder.image(processed_frame, channels="BGR")
        processed_video_writer.write(processed_frame)


    processed_video_writer.release()
    cap.release()
    os.remove(video_path)

    with open(processed_video_path, "rb") as file:
        # processed_video_file = base64.b64encode(file.read()).decode()
        processed_video_file = file.read()
        
    final_video_path = f'op-{video_name}.mp4'
    compress_video(processed_video_path, final_video_path)
    ffmpeg_command = f'ffmpeg -i {processed_video_path} -c:v libx264 -crf 23 {final_video_path} -y'
    # Run FFmpeg command
    res = subprocess.run(ffmpeg_command, shell=True)
    with open(final_video_path, "rb") as file:
        # final_video_file = base64.b64encode(file.read()).decode()
        final_video_file = file.read()
    
    os.remove(processed_video_path)
    return final_video_path, final_video_file





def main():
    st.title("Weapon Detection")
    st.link_button('github', 'https://github.com/jijnasu/Weapon-Detection')
    st.divider()

    model = get_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload Image or Video file for detection", type=["jpg", "jpeg", "png", "mp4"])
    # uploaded_file = 'inputs/images/gun.jpeg'
    if uploaded_file:
        st.divider()
        # st.header("Processing Result")
        # st.write(uploaded_file.name.split

        # Check if the uploaded file is an image or video
        file_name = os.path.splitext(uploaded_file.name)[0]
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        # file_name = 'gun'
        # file_extension = 'jpeg'

        if file_extension in ['.jpg', '.jpeg', '.png']:
            st.header('Uploaded Image')
            st.image(uploaded_file, caption=uploaded_file.name)
            st.header('Processing Result')
            processed_image = process_images(uploaded_file, model)
            output_file = f"op-{file_name}.jpg"
            st.image(processed_image, caption=output_file)
            download_button(processed_image, output_file, f"Download result {output_file}", key="processed_image")
        
        elif file_extension == '.mp4':
            st.header('Uploaded Video')
            st.video(uploaded_file, format="video/mp4", start_time=0)
            st.header('Processing Result')
            processed_video_path, processed_video_file = process_video(uploaded_file, model)
            # output_file = f"op-{file_name}.mp4"
            st.write('Final output video')
            st.video(processed_video_path, format="video/mp4", start_time=0)
            download_button(processed_video_file, processed_video_path, f"Download result {processed_video_path}", key="processed_video")
            os.remove(processed_video_path)
            # os.remove(video_path)

        else:
            st.warning("Unsupported file format. Please upload an image (jpg, jpeg, png) or video (mp4).")


    st.divider()
    st.subheader('About:')
    st.write('by Kumar Jijnasu')
    st.link_button('github.com/jijnasu','https://github.com/jijnasu')
    st.write('Other projects')
    st.link_button('Brain Tumor Classification','https://brain-tumor-classification-kj.streamlit.app')


if __name__ == "__main__":
    main()
