FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install deadsnakes and Python 3.11 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    screen \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ninja-build \
    libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# (Optional) Make python3 point to Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy fire
RUN python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

###
### Install OpenSSH server
###
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN echo "ClientAliveInterval 5" >> /etc/ssh/sshd_config
# Expose SSH port for the openssh server
EXPOSE 22

###
### Add non-root user with specific UID/GID
###
ENV USER_ID=2207
ENV GROUP_ID=2207

RUN groupadd -g ${GROUP_ID} user && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash user && \
    echo "user:password" | chpasswd && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

###
### Keep root for sshd, user available for SSH login or switching
###

USER root

CMD ["/usr/sbin/sshd", "-D"]
###
###
###