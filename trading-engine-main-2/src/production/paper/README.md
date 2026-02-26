Go to root directory of the project
```
docker build -t paper-service -f src/production/paper/Dockerfile .
```

```
docker run --rm -it \
  -v $(pwd)/service_account.json:/app/service_account.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/service_account.json" \
  paper-service
```