# Base image
FROM python:3.11-slim

# Install essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

COPY src/ src/
COPY data/ data/
COPY reports/ reports/

WORKDIR /

# ENTRYPOINT to execute evaluate.py
ENTRYPOINT ["python", "-u", "src/my_project/evaluate.py"]
