.PHONY: install reproduce offline test clean

install:
	python -m pip install -e ".[dev]"

reproduce:
	python scripts/run_all.py reproduce

offline:
	python scripts/run_all.py reproduce --offline

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

clean:
	rm -rf outputs/ data/external/ .pytest_cache/ src/*.egg-info
