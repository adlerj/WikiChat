.PHONY: install build run clean test

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Create virtual environment and install all dependencies
install: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install maturin
	cd crates/pocketwiki-python && ../../$(VENV)/bin/maturin develop
	$(PIP) install -e packages/pocketwiki-shared
	$(PIP) install -e packages/pocketwiki-builder
	$(PIP) install -e packages/pocketwiki-chat
	$(PIP) install -e "packages/pocketwiki-shared[dev]"
	$(PIP) install -e "packages/pocketwiki-builder[dev]"
	$(PIP) install -e "packages/pocketwiki-chat[dev]"
	touch $(VENV)/bin/activate

# Run the builder pipeline
build:
	$(VENV)/bin/pocketwiki-builder build --out /tmp/enwiki-bundle --source-url https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 $(ARGS)

# Run the chat service
run:
	$(VENV)/bin/pocketwiki-chat $(ARGS)

# Run tests
test:
	$(VENV)/bin/pytest

# Clean up
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
