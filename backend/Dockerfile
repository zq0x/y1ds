FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy your application code
COPY . /usr/src/app
WORKDIR /usr/src/app

# Expose the port your app runs on
EXPOSE 7861
EXPOSE 1370
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
EXPOSE 8003
EXPOSE 8004
EXPOSE 8005

# Command to run your application
CMD ["python3", "app.py"]
