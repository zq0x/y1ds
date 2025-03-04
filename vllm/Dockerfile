FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install vllm fastapi uvicorn pynvml

WORKDIR /app

COPY app.py /app/app.py

EXPOSE 1370

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1370"]