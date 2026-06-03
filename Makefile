# path: Makefile
.PHONY: install test lint clean build

install:
	pip install --upgrade pip
	pip install -e .[dev]

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

 test:
	pytest tests/ -v --tb=short

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

build: clean
	pip install build
	python3 -m build
