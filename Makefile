PYTHON ?= python
PYTHONPATH := $(CURDIR)
TMPDIR := $(CURDIR)/.tmp

.PHONY: test smoke

test:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m unittest discover -s tests -v

smoke:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" PYTHONPATH="$(PYTHONPATH)" $(PYTHON) -m unittest \
		tests.test_smoke -v
