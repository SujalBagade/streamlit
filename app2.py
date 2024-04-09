import cv2
import torch
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
import base64
import os

@st.cache_resource
def load_model():
    path = 'C:\yolov5safetyhelmet-main\yolov5safetyhelmet-main/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)
    #new_class_names = ['helmet', 'boots', 'safety-vests']
    #model.names = new_class_names
    return model

def object_detection(video_path, output_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 600))
        results = model(frame)
        frame = np.squeeze(results.render())
        results = model(frame)
        out.write(frame)
        cv2.imshow("FRAME", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    st.title("Object Detection on Video")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv", "webm"])

    if video_file is not None:
        temp_file = NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        temp_file.write(video_file.getvalue())
        temp_file.close()

        output_file = NamedTemporaryFile(delete=False, suffix=".avi")
        output_file_path = output_file.name
        output_file.close()

        st.write("Click the button to start object detection:")
        if st.button("Start Object Detection"):
            object_detection(temp_file_path, output_file_path)
            with open(output_file_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)

            # Download link for the output video
            binary_str = base64.b64encode(video_bytes).decode()
            href = f'<a href="data:video/avi;base64,{binary_str}" download="output_video.avi">Download Output Video</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()