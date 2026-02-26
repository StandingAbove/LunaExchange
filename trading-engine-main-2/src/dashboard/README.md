# Setup & Run Dashboard

### 1. Prerequisites

| Tool              | Version (or newer) | Notes                                  |
| ----------------- | ------------------ | -------------------------------------- |
| Docker Desktop    | 24 .x              | For the dashboard image                |
| (Optional) Python | 3.11               | Only if you want to run without Docker |
| (Optional) Poetry | ≥ 1.8              | Local-env route                        |


### 2. Run the research notebook in Docker (recommended)
   1. Point to your GCP service-account key (read-only access to the data bucket):
      ```
      export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
      ```
   2. Run Dashboard:
      ```
      cd src/dashboard       # contains docker-compose.yml
      docker compose up     # first run builds the image (~2-3 min)
      ```
   3. Open http://0.0.0.1:8501/ 

> To run outside of docker, run `python3 -m streamlit run ./src/dashboard/streamlit_entrypoint.py --server.port=8501 --server.address=0.0.0.0` from the root of the repo. Make sure your GCP credentials are exported as mentioned above.
