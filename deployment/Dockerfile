# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────
COPY app.py .

# ── Runtime ───────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
