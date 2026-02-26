FROM python:3.11-slim

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git \
 && curl -sSL https://install.python-poetry.org | python3 - \
 && ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY pyproject.toml poetry.lock* /work/
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction -E research --no-root

# Copy MOSEK license
RUN mkdir -p /root/mosek
COPY mosek.lic /root/mosek/mosek.lic
ENV MOSEKLM_LICENSE_FILE=/root/mosek/mosek.lic

RUN pip install jupyterlab ipywidgets matplotlib
ENV PYTHONPATH=/work/src

# Bring in the rest of the repo
COPY . /work

EXPOSE 8888
WORKDIR /work/src/research
CMD ["python","-m","jupyterlab", "--ip=0.0.0.0","--no-browser", "--ServerApp.token=","--ServerApp.password=", "--ServerApp.root_dir=/work/src/research", "--allow-root"]

