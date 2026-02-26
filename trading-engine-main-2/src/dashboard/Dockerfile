FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY src/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/common /app/src/common
COPY src/dashboard /app/src/dashboard

ENV PYTHONPATH="/app/"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_entrypoint.py", "--server.port=8501", "--server.address=0.0.0.0"]