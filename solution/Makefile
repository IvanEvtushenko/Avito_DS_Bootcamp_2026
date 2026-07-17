# Thin aliases over the CLI. Each target maps to one pipeline stage so that any
# stage can be run and inspected on its own (plan §5.1.4).
#
# Usage: make <target> [CONFIG=configs/default.yaml]

PYTHON ?= python3
CONFIG ?= configs/default.yaml
export PYTHONPATH := src

.PHONY: help test preprocess folds retrieve evaluate answer validate baseline clean

help:
	@echo "Targets:"
	@echo "  make test        - run unit + smoke tests (stdlib unittest, no GPU)"
	@echo "  make preprocess  - clean HTML corpus -> artifacts/corpus"
	@echo "  make folds       - dev/holdout split + 5-fold -> artifacts/splits"
	@echo "  make retrieve    - run configured retrievers -> artifacts/scores"
	@echo "  make evaluate    - MAP@10 / recall@k on calibration -> artifacts/runs"
	@echo "  make answer      - build answer.csv for test.f -> artifacts/answers"
	@echo "  make validate    - sanity-check answer.csv format"
	@echo "  make baseline    - full chain: preprocess -> folds -> retrieve -> evaluate -> answer -> validate"

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

preprocess:
	$(PYTHON) -m support_search.cli preprocess --config $(CONFIG)

folds:
	$(PYTHON) -m support_search.cli make-folds --config $(CONFIG)

retrieve:
	$(PYTHON) -m support_search.cli retrieve --config $(CONFIG)

evaluate:
	$(PYTHON) -m support_search.cli evaluate --config $(CONFIG)

answer:
	$(PYTHON) -m support_search.cli make-answer --config $(CONFIG)

validate:
	$(PYTHON) -m support_search.cli validate-answer --config $(CONFIG)

baseline: preprocess folds retrieve evaluate answer validate

clean:
	rm -rf artifacts/corpus artifacts/splits artifacts/scores artifacts/runs artifacts/answers
