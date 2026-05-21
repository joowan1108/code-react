FROM python:3.11-slim

LABEL org.opencontainers.image.title="team1351_v5" \
      org.opencontainers.image.version="v5" \
      org.opencontainers.image.description="KDD Cup 2026 Data Agents submission for team 1351 v5"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_COMPILE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py entrypoint.sh ./

RUN pip install --no-compile . \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
