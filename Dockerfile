# To build:
# docker buildx build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile.txt --tag lonce:transformerlw --load .


# Use latest NVIDIA PyTorch container
#FROM nvcr.io/nvidia/pytorch:25.01-py3
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
#FROM nvcr.io/nvidia/pytorch:22.05-py3

# Set up project directory
ARG PROJECT_DIR="/working"
WORKDIR "${PROJECT_DIR}"

# Ensure project directory exists
RUN mkdir -p "${PROJECT_DIR}"

# Use BuildKit caching for apt
RUN --mount=type=cache,target=/var/lib/apt \
    apt update && apt install -y git

# Install Python dependencies before copying all files (for better caching)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt



# Now copy the rest of the project
COPY . "${PROJECT_DIR}"

# Switch to final working directory
WORKDIR /transformerlw

# Create user and group safely
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN getent group ${GROUP_ID} || addgroup --gid ${GROUP_ID} user && \
    getent passwd ${USER_ID} || adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} user

USER user

# Set entrypoint
ENTRYPOINT ["/usr/bin/bash"]
