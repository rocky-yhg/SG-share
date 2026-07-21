PYTHON ?= python
PYTHONPATH := $(CURDIR)
TMPDIR := $(CURDIR)/.tmp

.PHONY: test smoke reproduce

test:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m unittest discover -s tests -v

smoke:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m sgshare.reproduce \
		--datasets synthetic --seeds 42 --output-root results/synthetic_matrix

reproduce:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m sgshare.reproduce \
		--datasets ces,globem --seeds 42,43,44 --data-root data/processed \
		--output-root results/reproduction --no-scaling
