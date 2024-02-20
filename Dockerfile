FROM openvino/ubuntu18_runtime

USER root

RUN apt-get update
RUN apt-get install build-essential cmake libboost-all-dev -y