FROM tensorflow/tensorflow:2.4.3-gpu

USER root
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 SHELL=/bin/bash

RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

RUN rm -rf /var/lib/apt/lists/* \
        /etc/apt/sources.list.d/cuda.list \
        /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils ca-certificates wget curl vim git openssh-server && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN python -m pip --no-cache-dir install -i https://mirrors.ustc.edu.cn/pypi/web/simple \
    Flask \
    Pillow

COPY . /mnist-serving/

CMD ['cd /mnist-serving; python serving.py']
