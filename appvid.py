import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64
from app import download_button, get_model

# Assuming you have a function to process a frame with YOLOv5
def process_frame(frame):
    # Your processing logic here
    # Replace this with your YOLOv5 model inference or any other processing
    # Process the first frame separately
    im_array = frame.plot()
    im = Image.fromarray(im_array[..., ::-1])
    frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return frame


def main():
    st.title("Video Processing App")
    model = get_model()
    # File uploader for video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video locally
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(uploaded_file.read())

        # Display the uploaded video
        video_file = open(video_path, "rb")
        st.video(video_file, format="video/mp4", start_time=0)

        # model inferance
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

        # Get video properties
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))

        # Create a video writer to save the processed video
        processed_video_path = 'processed_video.mp4'
        processed_video_writer = cv2.VideoWriter(
            processed_video_path,
            cv2.VideoWriter_fourcc(*'H264'),
            25,
            (frame_width, frame_height)
        )

        # Create an empty placeholder for the processed video
        processed_video_placeholder = st.empty()

        processed_frame = process_frame(first_result)
        processed_video_placeholder.image(processed_frame, channels="BGR")

        # Write the processed frame to the output video
        processed_video_writer.write(processed_frame)
        
        # Process each frame in the uploaded video
        # while True:
        for result in results:
            # ret, frame = cap.read()
            # if not ret:
            #     break

            # Convert BGR to RGB
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # # Process the frame
            # processed_frame_rgb = process_frame(frame_rgb)

            # # Convert RGB back to BGR
            # processed_frame = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)

            processed_frame = process_frame(result)
            # Display the processed frame
            processed_video_placeholder.image(processed_frame, channels="BGR")

            # Write the processed frame to the output video
            processed_video_writer.write(processed_frame)

        # Release the video writer and capture objects
        processed_video_writer.release()
        cap.release()

        # Provide a download link for the processed video
        download_link = create_download_link(processed_video_path, "Download Processed Video")
        st.markdown(download_link, unsafe_allow_html=True)

        # Remove the uploaded video file
        os.remove(video_path)

def create_download_link(file_path, link_text):
    with open(file_path, "rb") as file:
        encoded_file = base64.b64encode(file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{encoded_file}" download="{file_path}">{link_text}</a>'
    return href

if __name__ == "__main__":
    main()
