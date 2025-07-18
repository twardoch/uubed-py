# Makefile for uubed package
.PHONY: help build test lint clean install dev-install release test-release

help:
	@echo "Available targets:"
	@echo "  help         - Show this help message"
	@echo "  build        - Build the package"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  clean        - Clean build artifacts"
	@echo "  install      - Install package"
	@echo "  dev-install  - Install package in development mode"
	@echo "  release      - Create release (use VERSION=v1.2.3)"
	@echo "  test-release - Test release to TestPyPI"

build:
	python scripts/build.py

test:
	python scripts/test_runner.py

lint:
	ruff check src tests
	ruff format --check src tests

clean:
	rm -rf dist/ build/ *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache/ htmlcov/ .coverage

install:
	pip install .

dev-install:
	pip install -e .[test]

release:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION not set. Use: make release VERSION=v1.2.3"; \
		exit 1; \
	fi
	python scripts/release.py $(VERSION)

test-release:
	python scripts/release.py --test

# Development shortcuts
all: build test

ci: lint test

# Quick development iteration
dev: clean dev-install test