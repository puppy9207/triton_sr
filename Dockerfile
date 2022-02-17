FROM nvcr.io/nvidia/tritonserver:21.12-py3
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y python3-dev
RUN pip install opencv-python
