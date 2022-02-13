FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

ADD ./aliyun_source.list /etc/apt/sources.list

RUN apt-get update \
    && apt-get install -y build-essential strace \
    && apt-get install -y curl bzip2 wget zip unzip \
    && apt-get install -y pkg-config g++ zlib1g-dev vim \
    && apt-get install -y git openssh-server ssh jq nfs-common \
    && apt-get install -y tzdata locales \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get autoremove -y \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.utf8 LANGUAGE=en_US.utf8 LC_ALL=en_US.utf8

# Default install python 3.8
ARG PYTHON_VERSION=3.8

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /conda \
    && echo ". /conda/etc/profile.d/conda.sh" >> ~/.bashrc && echo "conda activate base" >> ~/.bashrc \
    && rm -rf /tmp/miniconda.sh

ENV PATH=/conda/bin:$PATH
RUN conda install pip \
    && pip install tensorflow==2.4.2 tensorflow_estimator==2.4.0 tensorflow-serving-api==2.4.0 \
    && pip install pytest==6.2.4 pyyaml==5.4.1 flake8

CMD [ "/bin/bash" ]

