FROM python:3.9-slim

RUN apt-get -y update
# Install FFmpeg and Cmake
RUN apt-get install -y ffmpeg \
    git \
    build-essential \
    cmake \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /FaceRecog

COPY ./ ./

RUN pip3 install -r requirements.txt

CMD ["bash"]