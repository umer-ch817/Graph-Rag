# T01 environment record

Inspected 2026-09-29. This record contains no credentials.

| Item | Observed value |
|---|---|
| Host | Windows x64 |
| Project interpreter | CPython 3.12.14, isolated `.venv` |
| Default system interpreter | CPython 3.14.3; not used for this project |
| CPU | Intel Core i5-10210U, 4 cores / 8 logical processors |
| Physical RAM | 8,362,000,384 bytes (approximately 7.79 GiB) |
| GPU | Intel UHD Graphics; CPU-first inference planned |
| Docker engine | 29.8.0, Linux containers through Docker Desktop / WSL2 |
| Dependency resolver | uv 0.12.15; hash-pinned `requirements-dev.lock` |
| Neo4j image | 5.26.31 Community (tag `5.26-community`, pinned digest below) |
| Container memory limit | 1,536 MiB; heap 256–512 MiB, page cache 256 MiB |

Registry manifest digest: `sha256:5eb12ad77fa46ab73e23df9ea1f43f5c0f2a79523435577648e046be042b9b93`.

Linux amd64 manifest: `sha256:e885941da359002feb6a61b607d4a3808b6dee7545364a2c50df56fb09262ec2`.

Registry inspection resolved the tag to 5.26.31 on this date. Compose pins the multi-platform digest, so later tag movement cannot change the image selected by this project.

## Verification

T01 completed. Models and paid LLM calls are outside T01; none were used.

- Fresh Python 3.12 virtual environment installed `requirements-dev.lock` using `pip install --require-hashes`, then the editable project using `--no-deps --no-build-isolation`.
- Configuration suite: **31 passed**, including a run in the fresh environment. Validates local URI restrictions, model ids, budget limits, environment-over-YAML precedence, no implicit dotenv read, unknown fields and secret redaction.
- `python -m ruff check .`: passed. `python -m ruff format --check src tests`: passed.
- `python -m compileall -q src`: passed. `python -m pip check`: no broken requirements.
- `docker compose config --quiet`: passed. `docker compose ps`: healthy, host bindings `127.0.0.1:7474` and `127.0.0.1:7687`.
- Neo4j Python driver authenticated, verified connectivity and returned `1` for `RETURN 1 AS healthy`; server reported `Neo4j/5.26.31`.
- Git ignore checks confirmed local credentials, virtual environments and generated smoke artifacts are excluded. Repository initialized locally; no commit or push performed.

The smoke environment is under ignored `artifacts/t01-smoke`. A random local database credential is stored in ignored `.env`; it is not included in this record. Neo4j is left running and can be stopped with `docker compose stop neo4j` without deleting its volume.

## Setup findings

Docker was installed but stopped; starting Docker Desktop restored the engine. The initial dependency tool spent time hashing unrelated platform distributions; the installed uv resolver produced the final hash lock. The clean-install check exposed Hatchling's editable-build dependency; `editables` is now explicitly included in the development lock so the documented no-build-isolation installation succeeds.
