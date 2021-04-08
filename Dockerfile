FROM nvidia/cuda:11.1-base

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install python3.7
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Source code
COPY . /app

RUN ["chmod", "+x", "/app/muzero.py"]

VOLUME ["C:\Users\hadis\Documents\docker_vol"]

CMD python3 /app/muzero.py heist
