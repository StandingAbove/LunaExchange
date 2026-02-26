# Setup & Run Research Environment

### 1. Prerequisites

| Tool              | Version (or newer) | Notes                                  |
| ----------------- | ------------------ | -------------------------------------- |
| Docker Desktop    | 24 .x              | For the one-command notebook image     |
| (Optional) Python | 3.11               | Only if you want to run without Docker |
| (Optional) Poetry | ≥ 1.8              | Local-env route                        |


### 2. Run the research notebook in Docker (recommended)
   1. Point to your GCP service-account key (read-only access to the data bucket):
      ```
      export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
      ```
   2. Fire up JupyterLab:
      ```
      cd src/research       # contains docker-compose.yml
      docker compose up     # first run builds the image (~2-3 min)
      ```
   3. Open http://localhost:8888 – no token required – and launch 00_quickstart.ipynb (already in /work/src/research).
