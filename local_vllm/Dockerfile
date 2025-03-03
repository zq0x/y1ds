FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04



# Support pipefail so we don't build broken images
SHELL ["/bin/bash", "-c"]


# Ubuntu 24.04 requires this for compatibility with our /.launch script
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Don't ask questions we cannot answer during the build
ENV DEBIAN_FRONTEND=noninteractive
# Allow immediate output
ENV PYTHONUNBUFFERED=1

# Interactive container
RUN yes | unminimize

# Create a useful base environment with commonly used tools
ARG TARGETARCH=amd64

RUN \
    set -euo pipefail && \
    ([ $TARGETARCH = "arm64" ] && echo "Skipping i386 architecture for ARM builds" || dpkg --add-architecture i386) && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
        software-properties-common \
        gpg-agent && \
    # For alternative Python versions
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        # Base system utilities
        acl \
        ca-certificates \
        locales \
        lsb-release \
        curl \
        wget \
        sudo \
        moreutils \
        nano \
        vim \
        less \
        jq \
        git \
        git-lfs \
        man \
        tzdata \
        # Display
        fonts-dejavu \
        fonts-freefont-ttf \
        fonts-ubuntu \
        ffmpeg \
        libgl1-mesa-glx \
        # System monitoring & debugging
        htop \
        iotop \
        strace \
        libtcmalloc-minimal4 \
        lsof \
        procps \
        psmisc \
        nvtop \
        # Development essentials
        build-essential \
        cmake \
        ninja-build \
        gdb \
        # System Python
        python3-full \
        python3-dev \
        python3-pip \
        # Network utilities
        netcat \
        net-tools \
        dnsutils \
        iproute2 \
        iputils-ping \
        traceroute \
        # File management
        rsync \
        rclone \
        zip \
        unzip \
        xz-utils \
        zstd \
        # Performance analysis
        linux-tools-common \
        # Process management
        supervisor \
        cron \
        # Required for cron logging
        rsyslog \
        # OpenCL General
        clinfo \
        pocl-opencl-icd \
        opencl-headers \
        ocl-icd-dev \
        ocl-icd-opencl-dev && \
    # Ensure TensorRT where applicable
    if [ -n "${CUDA_VERSION:-}" ]; then \
        CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d ".") && \
        if [ "$CUDA_MAJOR_MINOR" -ge "126" ]; then \
            apt-get update && apt-get install -y --no-install-recommends \
                libnvinfer10 \
                libnvinfer-plugin10; \
        elif [ "$CUDA_MAJOR_MINOR" -ge "121" ]; then \
            apt-get update && apt-get install -y --no-install-recommends \
                libnvinfer8 \
                libnvinfer-plugin8; \
        fi \
    fi
    # Install OpenCL Runtimes
    ARG TARGETARCH=amd64
    RUN \
    set -euo pipefail && \
        if command -v rocm-smi >/dev/null 2>&1; then \
            apt-get install -y rocm-opencl-runtime; \
        elif [ -n "${CUDA_VERSION:-}" ]; then \
            CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d ".") && \
            # Refer to https://docs.nvidia.com/deploy/cuda-compatibility/#id3 and set one version below to avoid driver conflicts (patch versions > driver)
            # Avoid transitional packages - They will cause NVML errors and broken nvidia-smi by bumping the version 
            if [ "${CUDA_MAJOR_MINOR}" -ge 118 ]; then \
                case "${CUDA_MAJOR_MINOR}" in \
                    "118"|"120") \
                        driver_version=470 \
                        ;; \
                    "121") \
                        driver_version=525 \
                        ;; \
                    "122") \
                        driver_version=525 \
                        ;; \
                    "123") \
                        driver_version=535 \
                        ;; \
                    "124") \
                        driver_version=535 \
                        ;; \
                    "125") \
                        driver_version=545 \
                        ;; \
                    "126") \
                        driver_version=555 \
                        ;; \
                    *) \
                        driver_version=555 \
                        ;; \
                esac \
            else \
                driver_version=390; \
            fi && \
            if [ "${TARGETARCH}" = "arm64" ] && [ "${driver_version}" -lt 510 ]; then \
                echo "No suitable libnvidia-compute package is available for arm64 with driver ${driver_version}"; \
            else \
                apt-get install -y libnvidia-compute-${driver_version}; \
            fi \
        fi && \
        apt-get clean -y
    

    
# Set up Python and pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app

# Expose the port your app runs on
EXPOSE 1370
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
EXPOSE 8003
EXPOSE 8004
EXPOSE 8005

# Command to run your application
CMD ["python3", "app.py"]