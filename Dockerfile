FROM mcr.microsoft.com/dotnet/sdk:10.0 AS api-build
WORKDIR /src
COPY api/ResumeRoleAnalyzer.Api.csproj api/
RUN dotnet restore api/ResumeRoleAnalyzer.Api.csproj
COPY api/ api/
RUN dotnet publish api/ResumeRoleAnalyzer.Api.csproj -c Release -o /publish --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY ai/ /tmp/ai/
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir /tmp/ai \
    && rm -rf /tmp/ai
COPY --from=api-build /publish/ api/
COPY datasets/ datasets/
COPY models/ models/
COPY web/ web/
RUN /opt/venv/bin/build-catalog-ranker \
    && chown -R app:app /app /opt/venv
ENV ASPNETCORE_ENVIRONMENT=Production \
    ASPNETCORE_HTTP_PORTS=8080 \
    Ai__PythonExecutable=/opt/venv/bin/python \
    Ai__JobsPath=/app/datasets/raw/job_positions.csv \
    Ai__TimeoutSeconds=30
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
ENTRYPOINT ["dotnet", "/app/api/ResumeRoleAnalyzer.Api.dll"]
