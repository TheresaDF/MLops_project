# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY vae_cats/ vae_cats/
COPY conf/ conf/
COPY .dvc/ .dvc/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN pip install dvc 
RUN dvc pull 

ENTRYPOINT ["python", "-u", "vae_cats/train_model.py"]