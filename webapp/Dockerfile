FROM python:3.8
RUN apt-get update && apt-get -y install cmake ffmpeg libsm6 libxext6

#requirements
COPY ar-ctrgcn-webapp/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

RUN pip3 install torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch==2.0.1
RUN pip3 install torchvision
RUN pip3 install --upgrade mediapipe
RUN pip3 install pandas psutil pyyaml tqdm ultralytics

#start
COPY ar-ctrgcn-webapp /ar-ctrgcn-webapp
WORKDIR /ar-ctrgcn-webapp
EXPOSE 5000:5000
ENTRYPOINT ["python3", "ar-ctrgcn-app.py"]