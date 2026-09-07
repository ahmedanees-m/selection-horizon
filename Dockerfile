FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        zlib1g-dev \
        libcurl4-openssl-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e ".[dev]"

COPY config ./config
COPY tests ./tests
COPY Makefile ./

ENV SH_DATA_ROOT=/work/data

CMD ["python", "-c", "import src; print('selection-horizon image ready')"]
