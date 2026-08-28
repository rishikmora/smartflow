# syntax=docker/dockerfile:1
#
# Build context is ./services (not this directory) so the shared common/ package
# is reachable. See docker-compose.yml.
FROM python:3.11-slim

# The services only ever read a mounted volume, so there is no reason to be root.
RUN useradd --create-home --uid 10001 smartflow
WORKDIR /app

COPY llm_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared layer first so a change to one service does not bust its cache.
COPY common ./common
COPY llm_service/main.py .

USER smartflow
ENV PYTHONUNBUFFERED=1 SMARTFLOW_DATA_ROOT=/data
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3   CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
