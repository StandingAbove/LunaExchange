FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY src/production/paper/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MOSEK license
RUN mkdir -p /root/mosek
COPY mosek.lic /root/mosek/mosek.lic
ENV MOSEKLM_LICENSE_FILE=/root/mosek/mosek.lic

# Copy source code
COPY src/production/paper /app/production/paper
COPY src/production/simulations/config.yaml /app/production/simulations/config.yaml
COPY src/common /app/common
COPY src/trading_engine /app/trading_engine

ENV PYTHONPATH=/app/

CMD ["python", "production/paper/main.py"]