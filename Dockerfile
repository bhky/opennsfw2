FROM python:3.12-slim-bookworm

RUN rm -f /etc/apt/apt.conf.d/docker-clean

# libGL is needed to avoid "ImportError: libGL.so.1" in OpenCV
# libglib2.0-0 is needed to avoid "ImportError: libgthread-2.0.so.0" in OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy the entire project.
COPY . .

# Create non-root user.
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Pre-download model weights to avoid download during runtime.
# Weights path: /home/appuser/.opennsfw2/weights/open_nsfw_weights.h5
RUN python -c "import opennsfw2; opennsfw2.make_open_nsfw_model()" || echo "Model download failed, will retry at runtime"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]