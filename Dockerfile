# Web demo by default; the same image runs the CLI for files and RTSP streams:
#   docker build -t escalator-monitor .
#   docker run -p 7860:7860 escalator-monitor
#   docker run -v "$PWD/data:/data" escalator-monitor \
#       escalator-monitor run --input /data/clip.mp4 --roi 812,140,1105,140,1290,1040,640,1040 \
#       --headless --output-dir /data/out
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# libGL/glib are needed by the opencv-python wheel that Ultralytics depends on.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch: roughly 1.5 GB smaller than the default CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY escalator_monitor ./escalator_monitor
RUN pip install ".[web]"
COPY app.py ./
COPY examples ./examples

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# Bake the detector weights into the image so the first request does not download them.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

EXPOSE 7860
CMD ["python", "app.py"]
