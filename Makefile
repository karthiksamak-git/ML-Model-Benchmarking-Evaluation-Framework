PY ?= .venv/bin/python
MLBE ?= $(PY) -m mlbe.cli

.PHONY: install test lint benchmark quick hpo analyze rubric repro export figures all web-install web-build clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

quick:
	$(MLBE) all --quick

benchmark:
	$(MLBE) benchmark

hpo:
	$(MLBE) hpo

analyze:
	$(MLBE) analyze

rubric:
	$(MLBE) rubric

repro:
	$(MLBE) repro --sample

export:
	$(MLBE) export

figures:
	$(MLBE) figures

all:
	$(MLBE) all

web-install:
	cd web && npm install

web-build:
	cd web && npm run build

web-test:
	cd web && npm run test:golden

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ web/.next
