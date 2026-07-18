# Deployment

The application is packaged as one container containing ASP.NET Core, the Python
analyzer, the O*NET catalog, and model artifacts.

## Local production check

```bash
docker compose build
docker compose up -d
curl http://localhost:8080/health
docker compose logs -f rolewise
```

Open `http://localhost:8080`. Stop it with `docker compose down`.
Docker Desktop or another Docker Engine with Compose support must be installed first.

## Hosting requirements

Deploy the built image to a container host that supports:

- at least 1 GB RAM and 1 shared CPU;
- an HTTPS reverse proxy forwarding to container port 8080;
- a writable temporary directory for uploads;
- health checks against `GET /health`;
- request bodies of at least 6 MB.

Set `ASPNETCORE_FORWARDEDHEADERS_ENABLED=true` when the hosting platform terminates
HTTPS at a trusted reverse proxy. Do not expose the container directly without HTTPS.

The API limits each client IP to 20 analysis requests per minute, accepts resumes up to
5 MB, times analysis out after 30 seconds, removes temporary uploads, and returns generic
production errors. Logs may contain analyzer diagnostics, so route them only to an
access-controlled service and configure an appropriate retention period.

## Updating data or models

Before rebuilding an image after catalog changes, run:

```bash
source ai/.venv/bin/activate
import-onet-catalog
train-resume-matcher
build-catalog-ranker
evaluate-career-corpus
```

Model files are ignored by Git in this starter project. Store production model artifacts
in a private release-artifact bucket or use Git LFS, then ensure they are present in
`models/` before building. The container downloads MiniLM and rebuilds the
sentence-embedding catalog artifact during the image build.
