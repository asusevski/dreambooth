FROM pytorch/pytorch:latest as base
RUN python3 -m pip install --upgrade pip
RUN apt-get update 

RUN apt-get install -y --fix-missing \
    git build-essential \
    cmake software-properties-common ffmpeg libsm6 libxext6 vim

COPY . /dreambooth
WORKDIR /dreambooth

RUN python3 -m pip install -r ./docker/requirements.txt

CMD ["bash"]
