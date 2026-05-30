VENV    ?= venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

.PHONY: help install ingest ingest-pilot resume reset-ingest serve test health clean-pycache

help:
	@echo "Common targets:"
	@echo "  make install        Install requirements into ./venv"
	@echo "  make ingest         Run the ingest (idempotent; skips URLs already indexed)"
	@echo "  make ingest-pilot   Ingest only the first 5 URLs (sanity check)"
	@echo "  make resume         Alias for 'make ingest' (quota-safe top-up)"
	@echo "  make reset-ingest   Wipe the Chroma collection and re-ingest from scratch"
	@echo "  make serve          Start uvicorn on http://127.0.0.1:8000"
	@echo "  make test           Run pytest"
	@echo "  make health         curl /health on a running server"

install:
	$(PIP) install -r requirements.txt

ingest:
	$(PY) -m ingest.ingest

ingest-pilot:
	$(PY) -m ingest.ingest --limit 5

# `resume` is the daily top-up command. The ingest checks the Chroma collection
# for each URL's first chunk id (url#0) and skips it if already present, so this
# will NOT re-embed any of the pages already indexed.
resume: ingest

reset-ingest:
	$(PY) -m ingest.ingest --reset

serve:
	$(UVICORN) app.main:app --reload --host 127.0.0.1 --port 8000

test:
	$(VENV)/bin/pytest -q

health:
	curl -fsS http://127.0.0.1:8000/health && echo

clean-pycache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
