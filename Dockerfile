# Doppler acoustic geolocator: FastAPI backend + static frontend.
# ffmpeg/ffprobe are included so MP3/M4A/FLAC/OGG decoding and recording
# start-time metadata (creation_time) work server-side.
#
#   docker build -t doppler .
#   docker run -p 8000:8000 -v doppler-data:/data doppler
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DOPPLER_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/doppler
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY app ./app
COPY run.py ./

RUN useradd --create-home --uid 10001 doppler \
    && mkdir -p /data \
    && chown doppler /data
USER doppler
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
